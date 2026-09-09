"""Verify endpoint telemetry in a persistent-call benchmark JSON artifact."""
import argparse
import json
from pathlib import Path


def verify(report):
    checks=[]
    for peer in report['peers']:
        final=peer['boundary_final']
        checks.append(bool(peer['transport_pass']) and final['returning_idle'] is False)
        events=final['boundary_events']
        canonical=set()
        completed=[t for t in peer['server']['turns'] if t['status']=='complete']
        checks.append(bool(completed))
        for turn in completed:
            for kind in ('idle_to_speech','speech_to_idle'):
                frames=[e for e in events if e['turn_id']==turn['id'] and e['kind']==kind
                        and e['status']=='frame']
                good=bool(frames) and [e['index'] for e in frames]==list(range(frames[0]['frames']))
                if good:
                    good=(frames[0]['alpha']==0 and frames[-1]['alpha']==1
                          and all(e.get('exact') is True and e['expected_sha256']==e['output_sha256']
                                  for e in (frames[0],frames[-1]))
                          and all(b['pts']>a['pts'] for a,b in zip(frames,frames[1:])))
                    if kind=='speech_to_idle':
                        canonical.add(frames[-1]['output_sha256'])
                checks.append(good)
        checks.append(len(canonical)==1)
    return dict(passed=bool(checks) and all(checks), checks=len(checks),
                failed=sum(not c for c in checks),
                scope='Sender telemetry and local transport; not receiver pixel equality or visual quality')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report',type=Path)
    args=parser.parse_args()
    result=verify(json.loads(args.report.read_text()))
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['passed'] else 1)
