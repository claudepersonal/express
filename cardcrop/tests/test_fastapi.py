import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time
import unittest

import cv2
import numpy as np
from fastapi.testclient import TestClient


class FastAPIIntegrationTests(unittest.TestCase):
    def test_real_upload_edits_enhance_and_registry_recovery(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(__file__).resolve().parents[1]
            old=os.environ.get('STORAGE_DIR')
            os.environ['STORAGE_DIR']=folder
            try:
                spec=importlib.util.spec_from_file_location('cardchop_fastapi_test',root/'backend/app.py')
                module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            finally:
                if old is None:os.environ.pop('STORAGE_DIR',None)
                else:os.environ['STORAGE_DIR']=old
            image=np.full((80,120,3),220,np.uint8)
            ok,encoded=cv2.imencode('.png',image);self.assertTrue(ok)
            with TestClient(module.app) as client:
                health=client.get('/health').json()
                self.assertEqual(health['backend'],'fastapi');self.assertEqual(health['frontend'],'vite')
                response=client.post('/api/cards/upload',files=[('files',('duplicate.png',encoded.tobytes(),'image/png')),('files',('duplicate.png',encoded.tobytes(),'image/png'))])
                self.assertEqual(response.status_code,200,response.text)
                a,b=response.json()['cards'];self.assertNotEqual(a['id'],b['id'])
                def wait(card_id):
                    deadline=time.time()+30
                    while time.time()<deadline:
                        card=client.get('/api/cards/'+card_id).json()['card']
                        if card['status'] not in ('processing','enhancing'):return card
                        time.sleep(.05)
                    self.fail('Worker did not finish')
                wait(a['id']);wait(b['id'])
                base='/api/cards/'+a['id']
                edit={'expectedRevision':0,'workbench':{'rotation':90,'quad':[[0,0],[1,0],[1,1],[0,1]],'customSettings':{'scale':1,'autoRotate':False,'sharpen':0}}}
                saved=client.patch(base+'/workbench',json=edit)
                self.assertEqual(saved.status_code,200,saved.text)
                self.assertEqual(saved.json()['card']['workbenchRevision'],1)
                self.assertEqual(client.patch(base+'/workbench',json=edit).status_code,409)
                self.assertEqual(client.post(base+'/enhance?scale=NaN').status_code,400)
                started=client.post(base+'/enhance');self.assertEqual(started.status_code,200,started.text)
                completed=wait(a['id'])
                self.assertEqual(completed['status'],'enhanced',completed)
                self.assertEqual(completed['metadata']['output_size'],'80x120')
                self.assertEqual(completed['metadata']['enhancement']['detection_mode'],'manual_quad')
                png=client.get(completed['enhancedUrl'])
                self.assertEqual(png.status_code,200);self.assertTrue(png.content.startswith(b'\x89PNG\r\n\x1a\n'))
                self.assertEqual(client.get('/api/not-real').status_code,404)
                self.assertEqual(client.post('/api/cards/upload',files={'files':('bad.png',b'not an image','image/png')}).status_code,400)
            records=json.loads((Path(folder)/'jobs.json').read_text())
            self.assertEqual(len(records),2)
            self.assertEqual(next(c for c in records if c['id']==a['id'])['workbench']['rotation'],90)
            module.jobs.clear()
            with TestClient(module.app) as client:
                recovered=client.get(base).json()['card']
                self.assertEqual(recovered['status'],'enhanced');self.assertEqual(recovered['workbenchRevision'],1)


if __name__=='__main__':unittest.main()
