"""Local persistent-peer measurements; does not start/stop services."""
import asyncio
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from soulx_rtc.benchmark_calls import run
from soulx_rtc.verify_boundaries import verify

ROOT=Path(__file__).resolve().parent

async def main():
    summaries=[]
    avatars=['portrait:girl','portrait:man','portrait:yongen','default']
    for label,ids,sessions,speakers,interrupt in [
        (name.split(':')[-1],[name],1,1,name=='default') for name in avatars
    ]+ [('ten-connected',avatars,10,1,False)]:
        output=ROOT/f'call-{label}.json'
        args=SimpleNamespace(url='http://127.0.0.1:1111',sessions=sessions,speakers=speakers,
            avatars=ids,turns=2,duration_seconds=0,
            audio=['/workspace/SoulX-FlashHead/benchmarks/comparison-10s.wav'],audio_seconds=5,
            gap=.5,idle_seconds=1,timeout=90,record=True,compact_evidence=True,
            snapshots_every=0,interrupt=interrupt,interrupt_after=.6,resume_after_interrupt=interrupt,
            output=str(output))
        print('START',label,flush=True)
        await run(args)
        report=json.loads(output.read_text())
        # Idle-only peers have no turn boundaries to verify.
        boundaries=verify({'peers':report['peers'][:speakers]})
        summary=dict(label=label,boundaries=boundaries,speaking_peers_checked=speakers,
            all_transport_pass=report['all_transport_pass'],cleanup_pass=report['cleanup_pass'],
            underruns=report['total_underrun_frames'],
            peers=[dict(avatar=p['avatar_id'],wire_fps=p['wire_fps'],prepare_ms=p['call_prepare_ms'],
                first_media_s=[t['first_media_s'] for t in p['server']['turns'] if t['first_media_s'] is not None],
                stage_ms=[t['stage_ms'] for t in p['server']['turns']],
                held_frames=p['server']['held_frames'],missed_slots=p['server']['missed_video_slots'],
                idle_decode_mean_ms=p['server']['idle_decode_mean_ms'],
                encoders=p['server']['video_encoders']) for p in report['peers']])
        summaries.append(summary)
        (ROOT/'live-summary.json').write_text(json.dumps(summaries,indent=2))
        print(json.dumps(summary),flush=True)
        if not boundaries['passed']:
            raise RuntimeError('Boundary verification failed; inspect saved evidence')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',required=True)
    args=parser.parse_args()
    ROOT=Path(args.output_dir)
    ROOT.mkdir(parents=True,exist_ok=True)
    asyncio.run(main())
