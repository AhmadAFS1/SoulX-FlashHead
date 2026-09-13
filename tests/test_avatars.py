from types import SimpleNamespace
import pytest
from soulx_rtc.avatars import AvatarCatalog


def test_catalog_only_exposes_approved_idle_ids(tmp_path):
    root=tmp_path/'banks'
    bank=root/'alice'/'certified'
    bank.mkdir(parents=True)
    (bank/'active_listening.mp4').touch()
    outside=tmp_path/'outside.mp4'
    outside.touch()
    bad=root/'escape'/'certified'
    bad.mkdir(parents=True)
    (bad/'active_listening.mp4').symlink_to(outside)
    catalog=AvatarCatalog(SimpleNamespace(avatar_root=str(root),idle_video=None,default_avatar_images=False))
    assert [e['id'] for e in catalog.public()]==['default','bank:alice']
    assert all('path' not in e for e in catalog.public())
    assert catalog.resolve('bank:alice')['path']==str(bank/'active_listening.mp4')
    for value in ('../../etc/passwd',str(outside),'unknown',[],None):
        with pytest.raises(ValueError):
            catalog.resolve(value)
