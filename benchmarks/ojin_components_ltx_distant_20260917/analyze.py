"""CPU-only sampled sharpness statistics; these do not score dental anatomy."""
import json
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent
TIMES=(.5,1.5,2.5,3.5,4.5,6.5,7.5)
NAMES=['LTX23-distant-indian-man-bicubic2-480x832-to-960x1664',
       'LTX23-distant-indian-man-legacy-general4x-resized2x-480x832-to-960x1664',
       'LTX23-distant-indian-man-FaceLandmarker-native2x-TRT-480x832-to-960x1664']


def main():
    points=json.loads((ROOT/NAMES[-1]/'mouth-landmarks.json').read_text())
    result={'execution':'CPU analysis of seven pre-encode PNG samples per treatment',
        'warning':'Sobel/Laplacian values measure local contrast, not correct teeth, lip-sync, or temporal stability','treatments':{}}
    images={}
    for name in NAMES:
        rows=[];images[name]=[]
        for timestamp in TIMES:
            rgb=cv2.cvtColor(cv2.imread(str(ROOT/name/f'frame-{timestamp:.1f}s.png')),cv2.COLOR_BGR2RGB)
            marks=np.asarray(points[round(timestamp*24)],np.float32)*2
            lo=marks.min(0);hi=marks.max(0);cx,cy=(lo+hi)/2
            width=max(16.,hi[0]-lo[0]);height=max(12.,hi[1]-lo[1])
            x0=max(0,round(cx-width*.75));x1=min(rgb.shape[1],round(cx+width*.75))
            y0=max(0,round(cy-height*1.4));y1=min(rgb.shape[0],round(cy+height*1.4))
            crop=rgb[y0:y1,x0:x1];gray=cv2.cvtColor(crop,cv2.COLOR_RGB2GRAY).astype(np.float32)
            gx=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3);gy=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)
            rows.append({'time_s':timestamp,'box':[x0,y0,x1,y1],
                'sobel_mean':float(np.hypot(gx,gy).mean()),
                'laplacian_variance':float(cv2.Laplacian(gray,cv2.CV_32F).var())})
            images[name].append(rgb)
        result['treatments'][name]={'samples':rows,
            'sobel_mean':float(np.mean([r['sobel_mean'] for r in rows])),
            'laplacian_variance_mean':float(np.mean([r['laplacian_variance'] for r in rows]))}
    base=images[NAMES[0]]
    for name in NAMES[1:]:
        result['treatments'][name]['full_frame_mae_vs_bicubic']=float(np.mean([
            np.abs(a.astype(np.float32)-b.astype(np.float32)).mean() for a,b in zip(images[name],base)]))
    (ROOT/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
