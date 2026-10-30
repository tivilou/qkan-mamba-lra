"""Engineering diagnostics; these tests do not certify a frozen baseline cohort."""
import copy
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.baseline_delivery.common import ROOT, atomic_json, child_spec, dictionary_path, digest, read
from experiments.baseline_delivery.data import Split
from experiments.baseline_delivery.models import Classifier, load_s4d, optimizer
from experiments.baseline_delivery.runner import check_release, deliver, discover_gpus, return_files
from experiments.baseline_delivery.sync import download
from experiments.baseline_delivery.worker import save_checkpoint, load_checkpoint

torch.set_num_threads(1)
CACHE=Path(os.environ['QKAN_TEST_CACHE'])


def config(task='image',model='transformer'):
    return dict(task=task,model=model,width=8,layers=1,heads=2,state_size=4,
                dropout=0.,lr=.001,weight_decay=.01,vocabulary=16 if task=='retrieval' else None,length=16)


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.target=Path(self.temp.name)/'object'
        self.body=b'pinned upstream bytes';self.sha=hashlib.sha256(self.body).hexdigest()
        body=self.body
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*_): pass
            def do_GET(self):
                if self.headers.get('Authorization')!='Bearer test-token':
                    self.send_response(401);self.end_headers();return
                if self.path=='/redirect':
                    self.send_response(302);self.send_header('Location','https://other.invalid/object');self.end_headers();return
                value=body if self.path=='/object' else b'changed upstream bytes'
                self.send_response(200);self.send_header('Content-Length',str(len(value)));self.end_headers();self.wfile.write(value)
        self.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.url='http://127.0.0.1:'+str(self.server.server_port)
    def tearDown(self):
        self.server.shutdown();self.server.server_close();self.thread.join();self.temp.cleanup()
    def fetch(self,route='/object',token='test-token'):
        download(self.url+route,self.target,len(self.body),self.sha,token=token,test_http=True)
    def test_verified_cache_survives_retry(self):
        self.fetch();self.fetch(token='invalid');self.assertEqual(self.target.read_bytes(),self.body)
    def test_wrong_hash_never_commits(self):
        with self.assertRaises(ValueError): self.fetch('/corrupt')
        self.assertFalse(self.target.exists())
    def test_no_redirect_or_auth_forward(self):
        with self.assertRaises(ValueError): self.fetch('/redirect')
        self.assertFalse(self.target.exists())
    def test_unauthenticated_download(self):
        with self.assertRaises(ValueError): self.fetch(token='invalid')
    def test_existing_changed_bytes_preserved(self):
        self.target.write_bytes(b'local evidence')
        with self.assertRaises(ValueError): self.fetch()
        self.assertEqual(self.target.read_bytes(),b'local evidence')


class RunnerTests(unittest.TestCase):
    def test_exact_child_command(self):
        spec=child_spec('--help')
        self.assertEqual(spec['argv'][1:4],['-B','-m','experiments.baseline_delivery.worker'])
        self.assertEqual(spec['cwd'],str(ROOT))
    def test_formal_gate_is_closed(self):
        with self.assertRaisesRegex(ValueError,'dataset qualification'): check_release()
    @patch('experiments.baseline_delivery.runner.subprocess.check_output')
    def test_gpu_selection_respects_visibility(self,command):
        command.return_value='0, GPU-a, GPU A, 12000\n1, GPU-b, GPU B, 12000\n'
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'GPU-b'}):
            self.assertEqual(discover_gpus('auto')[0],['1'])
            for value in ('0','1,1','2',''):
                with self.assertRaises(ValueError): discover_gpus(value)
    def test_streamed_chunk_reassembly(self):
        # The remote parent directory preserves each local part basename.
        (ROOT/'runs').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'runs') as tmp:
            path=Path(tmp)/'weights.pt';path.write_bytes(bytes(range(255))*17)
            mappings=return_files(path,'checkpoints/weights.pt',part_size=1024)
            record=read(Path(mappings[-1].split('=')[0]))
            parts=[Path(tmp)/'return-chunks'/p['name'] for p in record['parts']]
            self.assertEqual(b''.join(p.read_bytes() for p in parts),path.read_bytes())
            self.assertEqual(record['original_sha256'],digest(path))
            self.assertEqual(mappings,return_files(path,'checkpoints/weights.pt',part_size=1024))
    def test_checkpoint_corruption_rejected_previous_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            state={'schema':'qkan-training-state.v1','contract':'contract','step':1,'model':{'weight':torch.ones(2)}}
            save_checkpoint(root,state);first=read(root/'checkpoints/latest.json')['file']
            state['step']=2;save_checkpoint(root,state)
            pointer=read(root/'checkpoints/latest.json')
            self.assertTrue((root/'checkpoints'/first).exists())
            self.assertEqual(load_checkpoint(root,'contract')['step'],2)
            (root/'checkpoints'/pointer['file']).write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'Corrupt'): load_checkpoint(root,'contract')


class InputAdapterTests(unittest.TestCase):
    def test_curated_dictionary_preserves_nested_source_identity(self):
        path=dictionary_path('/dataset/v3','text')
        self.assertEqual(path,Path('/dataset/v3/metadata/source-v1/text-dictionary.json'))
    def test_integer_inputs_separate_documents_and_mask(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'retrieval').mkdir()
            a=np.ones((2,4000),dtype=np.uint16);b=a.copy();a[:,:2]=[4,5];b[:,:2]=[6,7]
            np.savez(root/'retrieval/train.npz',inputs1=a,inputs2=b,labels=np.array([0,1]))
            data=Split(root,'retrieval','train',root/'cache');batch=data.batch([0,1],'cpu')
            self.assertEqual(batch['inputs1'].dtype,torch.long)
            self.assertFalse(torch.equal(batch['inputs1'],batch['inputs2']))
            self.assertEqual(batch['inputs1'][0,2].item(),1)
    def test_pixel_conversion_exact_endpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'image').mkdir();x=np.zeros((2,1024),dtype=np.uint8);x[1]=255
            np.savez(root/'image/train.npz',inputs=x,labels=np.array([0,1]))
            batch=Split(root,'image','train',root/'cache').batch([0,1],'cpu')
            self.assertEqual(batch['inputs'].dtype,torch.float32)
            torch.testing.assert_close(batch['inputs'][:,0],torch.tensor([-.5,.5]))
    def test_padding_does_not_change_document_encoding(self):
        model=Classifier(config('retrieval'),CACHE).eval()
        with torch.no_grad():
            a=model.encode(torch.tensor([[4,5,1]]));b=model.encode(torch.tensor([[4,5,1,1,1]]))
        torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
    def test_pair_head_gradients_reach_both_documents(self):
        model=Classifier(config('retrieval'),CACHE)
        batch={'inputs1':torch.tensor([[4,5,1]]),'inputs2':torch.tensor([[6,7,1]])}
        logits=model(batch);logits.square().sum().backward()
        gradient=model.embedding.weight.grad
        self.assertGreater(gradient[4:8].abs().sum().item(),0)
        self.assertEqual(gradient[1].abs().sum().item(),0)
        model.eval()
        with torch.no_grad():
            torch.testing.assert_close(model(batch),model({'inputs1':batch['inputs2'],'inputs2':batch['inputs1']}))

    def test_trace_dtype_structure_values_and_corruption(self):
        from experiments.baseline_delivery.trace import capture,semantic_validate
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'image').mkdir()
            np.savez(root/'image/val.npz',inputs=np.zeros((2,1024),dtype=np.uint8),labels=np.array([0,1]))
            split=Split(root,'image','val',root/'cache')
            cfg={**config(model='s4d_minimal'),'source_revision':'unit-fixture'}
            model=Classifier(cfg,CACHE);out=root/'result';out.mkdir()
            capture(model,split,out,cfg,'0'*64)
            self.assertTrue(read(out/'trace-validation.json')['passed'])
            artifact=next((out/'trace-values').glob('*.npy'));artifact.write_bytes(b'corrupt')
            with self.assertRaises((ValueError,OSError)): semantic_validate(out/'sample-trace.json')


class SuiteReturnTests(unittest.TestCase):
    def test_real_http_return_retry_preserves_snapshot_and_excludes_private(self):
        from test_exchange import service
        from scripts import exchange_artifacts as client
        (ROOT/'runs').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'runs') as tmp:
            root=Path(tmp);suite=root/'suite';job=suite/'jobs/image-transformer-s42';(job/'checkpoints').mkdir(parents=True)
            atomic_json(suite/'suite.json',{'source_revision':'fixture','jobs':[{'id':job.name}]})
            atomic_json(suite/'status.json',{'job':{'state':'computed'}})
            atomic_json(suite/'compute-complete.json',{'state':'computed','jobs':1})
            for name,value in [('config.json',{}),('environment.json',{}),('summary.json',{'formal':True}),('trace-validation.json',{'passed':True})]:
                atomic_json(job/name,value)
            (job/'best-model.pt').write_bytes(b'weights');(job/'checkpoints/state.pt').write_bytes(b'full state')
            atomic_json(job/'checkpoints/latest.json',{'file':'state.pt'})
            (job/'sample-trace.json').write_text('PRIVATE TEXT MUST STAY LOCAL')
            storage=root/'remote';storage.mkdir()
            server=service.UploadServer(('127.0.0.1',0),service.UploadHandler,root=storage,
                web_root=ROOT/'tools/exchange/web',token='test-only-token',max_bytes=1024*1024,namespace='qkan-mamba')
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            try:
                transport=client.Transport('http://127.0.0.1:'+str(server.server_port),'test-only-token',allow_http=True)
                original=transport.request;failed=[False]
                def fail_one(method,route,body=None,size=0):
                    if method=='PUT' and not failed[0]:
                        failed[0]=True;raise OSError('simulated lost upload')
                    return original(method,route,body,size)
                transport.request=fail_one
                with patch.object(client,'Transport',return_value=transport):
                    with self.assertRaises(OSError): deliver(suite)
                    self.assertFalse((suite/'delivery-complete.json').exists())
                    atomic_json(suite/'status.json',{'changed_after_first_upload':True})
                    deliver(suite);deliver(suite)
                remote=storage/'qkan-mamba/baseline-suite/suite'
                self.assertEqual(read(remote/'return-status.json'),{'job':{'state':'computed'}})
                self.assertFalse(list(remote.rglob('sample-trace.json')))
                self.assertEqual(read(suite/'delivery-complete.json')['state'],'delivered')
                manifest=read(suite/'delivery.json')
                self.assertEqual(manifest['source_revision'],'fixture')
                self.assertTrue(all(r['status']=='verified' for r in manifest['files']))
            finally:
                server.shutdown();server.server_close();thread.join()


class PinnedS4DTests(unittest.TestCase):
    def test_fft_matches_direct_causal_convolution(self):
        torch.manual_seed(19)
        model=load_s4d(CACHE)(4,d_state=4,dropout=0.,transposed=True,lr=.001)
        x=torch.randn(2,4,9);k=model.kernel(9)
        direct=torch.zeros_like(x)
        for t in range(x.shape[-1]):
            for j in range(t+1): direct[:,:,t]+=x[:,:,j]*k[:,t-j]
        expected=model.output_linear(model.activation(direct+x*model.D[:,None]))
        torch.testing.assert_close(model(x)[0],expected,atol=1e-6,rtol=1e-5)
    def test_kernel_matches_independent_discrete_recurrence(self):
        model=load_s4d(CACHE)(4,d_state=4,dropout=0.)
        p=model.kernel;dt=p.log_dt.exp();a=-p.log_A_real.exp()+1j*p.A_imag
        c=torch.view_as_complex(p.C);ad=torch.exp(a*dt[:,None]);bd=torch.expm1(a*dt[:,None])/a
        state=bd;expected=[]
        for _ in range(9):
            expected.append(2*(c*state).sum(-1).real);state=ad*state
        torch.testing.assert_close(p(9),torch.stack(expected,-1),atol=1e-6,rtol=1e-5)
    def test_parameter_optim_metadata_preserved(self):
        model=Classifier(config(model='s4d_minimal'),CACHE);opt=optimizer(model,config(model='s4d_minimal'))
        for param in model.parameters():
            if hasattr(param,'_optim'):
                group=next(g for g in opt.param_groups if any(p is param for p in g['params']))
                for key,value in param._optim.items(): self.assertEqual(group[key],value)
    def test_import_only_adapter_matches_original_source_gradients(self):
        # Load verbatim upstream in its original module namespace, then compare
        # output/loss/all gradients/one optimizer update against import adaptation.
        import sys,types
        entry=read(ROOT/'experiments/baseline_delivery/assets.json')['objects'][0]
        folder=CACHE/'upstream'/entry['id']
        from importlib.util import spec_from_file_location,module_from_spec
        spec=spec_from_file_location('_original_dropout',folder/'dropout.py');drop=module_from_spec(spec);spec.loader.exec_module(drop)
        nn_module=types.ModuleType('src.models.nn');nn_module.DropoutNd=drop.DropoutNd
        with patch.dict(sys.modules,{'src.models.nn':nn_module}):
            original=types.ModuleType('_original_s4d');exec(compile((folder/'s4d.py').read_text(),str(folder/'s4d.py'),'exec'),original.__dict__)
        torch.manual_seed(23);left=original.S4D(4,d_state=4,dropout=0.,transposed=False)
        right=load_s4d(CACHE)(4,d_state=4,dropout=0.,transposed=False);right.load_state_dict(left.state_dict())
        x=torch.randn(2,9,4);a=left(x)[0];b=right(x)[0];torch.testing.assert_close(a,b,rtol=0,atol=0)
        a.square().mean().backward();b.square().mean().backward()
        for pa,pb in zip(left.parameters(),right.parameters()): torch.testing.assert_close(pa.grad,pb.grad,rtol=0,atol=0)
        torch.optim.AdamW(left.parameters(),lr=.001).step();torch.optim.AdamW(right.parameters(),lr=.001).step()
        for pa,pb in zip(left.parameters(),right.parameters()): torch.testing.assert_close(pa,pb,rtol=0,atol=0)


if __name__=='__main__': unittest.main()
