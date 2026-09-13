"""Server-owned avatar catalog. Clients select IDs, never filesystem paths."""
import io
import math
import numpy as np
from pathlib import Path

import av
from PIL import Image


def portrait_anchor(path,width,height):
    """Exact PIL branch of model center-crop, without importing Torch in RTP process."""
    with Image.open(path) as source:
        source=source.convert('RGB')
        scale=max(height/source.height,width/source.width)
        w,h=math.ceil(scale*source.width),math.ceil(scale*source.height)
        resized=source.resize((w,h),resample=Image.Resampling.BILINEAR)
        x,y=int(round((w-width)/2.)),int(round((h-height)/2.))
        return np.array(resized.crop((x,y,x+width,y+height)))


class AvatarCatalog:
    def __init__(self, args):
        default=getattr(args,'idle_video',None)
        self.entries={'default':dict(id='default',name='Default',path=default,
                                     kind='video' if default else 'image')}
        self.previews={}
        if getattr(args,'default_avatar_images',True):
            defaults=(('portrait:girl','Girl','examples/girl.png'),
                      ('portrait:man','Man','/workspace/MuseTalk/assets/demo/man/man.png'),
                      ('portrait:yongen','Yongen','/workspace/MuseTalk/assets/demo/yongen/yongen.jpeg'))
            for key,name,path in defaults:
                candidate=Path(path).resolve()
                if candidate.is_file():
                    self.entries[key]=dict(id=key,name=name,path=str(candidate),kind='image')
        root=Path(getattr(args,'avatar_root',None) or
                  '/workspace/MuseTalk/assets/ltx23_pose_banks').resolve()
        if root.is_dir():
            for bank in sorted(root.iterdir()):
                if not bank.is_dir() or not bank.resolve().is_relative_to(root):
                    continue
                for filename in ('idle_active_listening.mp4','active_listening.mp4','neutral_resting.mp4'):
                    path=bank/'certified'/filename
                    if path.is_file() and path.resolve().is_relative_to(root):
                        key='bank:'+bank.name
                        if default and path.resolve() == Path(default).resolve():
                            break
                        names={'sample_ai_human_facetime_v1':'Classic',
                               'sample_ai_human_facetime_v2_variable':'Variable',
                               'sample_ai_human_facetime_v3_hyperreal':'Hyperreal',
                               'sample_ai_human_facetime_v4_balanced':'Balanced',
                               'sample_ai_human_facetime_closeup_production_v1':'Close-up',
                               'sample_ai_human_facetime_wide_production_v1':'Wide'}
                        self.entries[key]=dict(id=key,name=names.get(bank.name,bank.name.replace('_',' ')),
                                               path=str(path.resolve()),kind='video')
                        break

    def public(self):
        return [dict(id=e['id'],name=e['name'],kind=e.get('kind','video'),
                     preview_url='/avatars/'+e['id']+'/preview.jpg')
                for e in self.entries.values()]

    def resolve(self, key):
        if not isinstance(key,str) or key not in self.entries:
            raise ValueError('Unknown avatar_id; choose an ID from /avatars')
        return self.entries[key]

    def preview(self, key):
        entry=self.resolve(key)
        if key not in self.previews:
            if entry['path'] and entry['kind']=='video':
                with av.open(entry['path']) as source:
                    rgb=next(source.decode(video=0)).reformat(width=192,height=192,format='rgb24').to_ndarray()
                image=Image.fromarray(rgb)
            elif entry['path']:
                with Image.open(entry['path']) as source:
                    image=source.convert('RGB').resize((192,192))
            else:
                with Image.open('examples/girl.png') as source:
                    image=source.convert('RGB').resize((192,192))
            output=io.BytesIO()
            image.save(output,format='JPEG',quality=85,optimize=True)
            self.previews[key]=output.getvalue()
        return self.previews[key]
