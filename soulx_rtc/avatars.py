"""Server-owned avatar catalog. Clients select IDs, never filesystem paths."""
from pathlib import Path


class AvatarCatalog:
    def __init__(self, args):
        default=getattr(args,'idle_video',None)
        self.entries={'default':dict(id='default',name='Server default avatar',path=default)}
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
                        self.entries[key]=dict(id=key,name=bank.name.replace('_',' '),path=str(path.resolve()))
                        break

    def public(self):
        return [dict(id=e['id'],name=e['name'],kind='video' if e['path'] else 'image')
                for e in self.entries.values()]

    def resolve(self, key):
        if not isinstance(key,str) or key not in self.entries:
            raise ValueError('Unknown avatar_id; choose an ID from /avatars')
        return self.entries[key]
