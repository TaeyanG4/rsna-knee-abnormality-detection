"""Render original, data-free diagrams for the English/Korean research snapshot.

No MRI, labels, reports or model files are read. The score/aggregate values below
are transcribed from docs/reports/project-summary-20261022.md (18:15 KST cutoff).
Requires numpy, matplotlib and Pillow. Run from the project root with Python.
"""
from pathlib import Path
import hashlib
import json
import io

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Ellipse, FancyArrowPatch
from matplotlib import font_manager
from PIL import Image
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/reports/visuals-20261022'
OUT.mkdir(parents=True, exist_ok=True)
font_path = Path('C:/Windows/Fonts/malgun.ttf')
if font_path.exists():
    font_manager.fontManager.addfont(str(font_path))
    plt.rcParams['font.family'] = 'Malgun Gothic'
else:
    plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams.update({'axes.unicode_minus': False, 'font.size': 12})
BG, INK, MUTED = '#F5F7FA', '#142A43', '#52657A'
BLUE, TEAL, ORANGE = '#286AC1', '#00867D', '#BD640E'
LINE, WHITE = '#D6DFE8', '#FFFFFF'

def canvas(title, subtitle, height=9):
    fig = plt.figure(figsize=(16, height), dpi=125, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set(xlim=(0, 1600), ylim=(height*100, 0)); ax.axis('off')
    ax.text(60, 55, 'RSNA KNEE  /  RESEARCH NOTES', color=TEAL, size=11, weight='bold')
    ax.text(60, 112, title, color=INK, size=27, weight='bold')
    ax.text(60, 152, subtitle, color=MUTED, size=12)
    return fig, ax

def box(ax, x, y, w, h, color=WHITE, edge=LINE, radius=18):
    p=FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={radius}',facecolor=color,edgecolor=edge,linewidth=1.1)
    ax.add_patch(p); return p

def text(ax,x,y,value,size=13,color=INK,weight='normal',**kw):
    return ax.text(x,y,value,size=size,color=color,weight=weight,va='top',**kw)

def arrow(ax,x1,y1,x2,y2,color=MUTED):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=17,color=color,linewidth=1.5))

def footer(ax, y, source, note):
    ax.plot([60,1540],[y,y],color=LINE,lw=1)
    text(ax,60,y+20,source,10,MUTED)
    text(ax,60,y+45,note,10,MUTED)

manifest=[]
def save(fig,name,description,sources):
    buf=io.BytesIO(); fig.savefig(buf,format='png',dpi=125,facecolor=BG)
    buf.seek(0)
    # Re-encode pixel-only PNG: strip all text/EXIF/software metadata.
    with Image.open(buf) as source:
        clean=Image.frombytes('RGB',source.size,source.convert('RGB').tobytes())
        path=OUT/name; clean.save(path,format='PNG',optimize=True)
    manifest.append({'file':name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size,'width':clean.width,'height':clean.height,'description':description,'sources':sources,'patient_data':False})
    plt.close(fig)

def slice_stack(ax,x,y,plane):
    # Pure vector slice icons, deliberately not anatomical MRI or model output.
    for k in range(3,-1,-1):
        sx,sy=x+k*9,y-k*7
        box(ax,sx,sy,145,88,'#20344A','#64788B',5)
        ax.add_patch(Ellipse((sx+73,sy+44),76,56,angle=20,facecolor='#A1ADBA',edgecolor='none'))
        ax.add_patch(Ellipse((sx+59,sy+37),33,34,angle=20,facecolor='#DBE2E8',edgecolor='none'))
        ax.add_patch(Ellipse((sx+85,sy+57),24,22,angle=20,facecolor='#DBE2E8',edgecolor='none'))
        ax.plot([sx+45,sx+101],[sy+49,sy+42],color='#20344A',lw=4)
    text(ax,x,y+109,plane,11,MUTED)

fig,ax=canvas('From MRI studies to 12 predictions','MRI 입력부터 12개 예측값까지 · conceptual diagram / 개념도',9)
box(ax,60,204,420,480)
text(ax,88,230,'01  MRI study',20,INK,'bold')
text(ax,88,273,'Multiple series, ordered slices',12,MUTED)
slice_stack(ax,98,341,'Sagittal / 시상면')
slice_stack(ax,276,341,'Coronal / 관상면')
slice_stack(ax,98,520,'Axial / 축상면')
text(ax,280,518,'Schematic icons\nnot patient MRI\n\n실제 환자 영상 아님',11,MUTED)
arrow(ax,491,445,529,445)
box(ax,540,204,450,480)
text(ax,568,230,'02  Render + model branches',20,INK,'bold')
text(ax,568,273,'Decode / orient / sample / normalize',12,MUTED)
box(ax,568,316,394,76,'#EDF4FD','#BDD2EE',12)
text(ax,586,332,'Shared R6 / K1 render where registered',13,BLUE,'bold')
text(ax,586,359,'Reuse input work; preserve serving parity',11,MUTED)
for y,title,detail,col in [(418,'P  ·  Public branch','R384 / R224 / R6flip',BLUE),(496,'O  ·  Own-model bundle','EXP143v2 / EXP155 / Q',ORANGE),(574,'K  ·  K1 family','K3 trio or K4 with full-bag member',TEAL)]:
    box(ax,568,y,394,62,WHITE,LINE,10);text(ax,584,y+9,title,13,col,'bold');text(ax,584,y+34,detail,10,MUTED)
arrow(ax,1001,445,1039,445)
box(ax,1050,204,490,480)
text(ax,1078,230,'03  Rank fusion + output',20,INK,'bold')
text(ax,1078,273,'Fixed recipe for each registered candidate',12,MUTED)
for i in range(12):
    x=1082+(i%6)*70;y=339+(i//6)*82
    box(ax,x,y,56,55,'#E4F3F0','#ADD8D1',10)
    ax.text(x+28,y+28,f'{i+1:02d}',ha='center',va='center',size=13,color=TEAL,weight='bold')
text(ax,1078,520,'One study ID × 12 finite target values',15,INK,'bold')
text(ax,1078,558,'Validate coverage, fallback behavior\nand actual member execution before submission.',12,MUTED,linespacing=1.8)
footer(ax,746,'Source: consolidated project report §3 and §5 · schematic, not a universal execution graph.','Only registered branches are used in each candidate. Tiles are original diagram artwork; no data or labels are embedded.')
save(fig,'study-to-prediction.png','Conceptual slice-stack and model-serving overview.',['project-summary-20261022.md §3','project-summary-20261022.md §5'])

fig,ax=canvas('How the candidate families differ','후보별 차이는 무엇인가 · registered structure at the 2026-10-22 18:15 KST cutoff',10)
for x,w,title,body,col in [(60,330,'P  Public','Target-specific public ratios\nFlat P sets four ratios to 0.5',BLUE),(410,330,'O  Own-model','Prior own bundle\nINF3 retains a 15% O share',ORANGE),(760,370,'K3  Trio','K1 v2 + v3-G seed0 + seed1\nRank aggregation of 3 members',TEAL),(1150,390,'K4  Four members','K3 + full-bag fold-0 member\nAn addition, not a seed0 replacement',TEAL)]:
    box(ax,x,205,w,142);text(ax,x+20,225,title,17,col,'bold');text(ax,x+20,267,body,11,MUTED,linespacing=1.8)
cards=[
 (60,393,'X','80% INF3 + 20% K3','K3 stride 3','INF3 = rank fusion of 85% P + 15% O',BLUE),
 (565,393,'p25 / F','75% P + 25% K3','K3 stride 2','F uses flat P; runtime v10',TEAL),
 (1070,393,'U','50% X_u + 50% Y_u','K3 stride 2','X_u: 80/20 INF3/K3; Y_u: 75/25 P/K3',BLUE),
 (60,644,'Rank operation  r(·)','Full-roster, per-target ranks','Average ties','Percentages here summarize nested rank fusion.',MUTED),
 (565,644,'P4 / F4','75% P + 25% K4','K4 stride 2','F4 uses flat P; preparation only at cutoff',TEAL),
 (1070,644,'U4','U recipe with K4','K4 stride 2','Keep the same two fusion branches.',BLUE),
]
for x,y,title,formula,detail,note,col in cards:
    box(ax,x,y,470,200);text(ax,x+23,y+22,title,23,col,'bold');text(ax,x+23,y+73,formula,16,INK,'bold');text(ax,x+23,y+112,detail,12,MUTED);text(ax,x+23,y+152,note,10,MUTED)
arrow(ax,800,599,800,637,TEAL);arrow(ax,1305,599,1305,637,BLUE)
footer(ax,889,'Source: project-summary-20261022.md §3. This diagram omits exact precision/rendering details; use versioned code and reviews.','X_u is a component of U, not the submitted X implementation. U/P4/U4 outcomes and F4 acceptance were unrecorded at cutoff.')
save(fig,'candidate-family-map.png','Registered model components and candidate-family contrasts.',['project-summary-20261022.md §3','D-B-278'])

fig,ax=canvas('Public snapshot — outcomes are still incomplete','공개 점수 현황 · Taeyang-supplied rounded displays, not private results',9)
box(ax,60,203,900,480);box(ax,990,203,550,480)
text(ax,89,228,'RECORDED PUBLIC DISPLAY',13,TEAL,'bold')
plot=fig.add_axes([.13,.34,.43,.34],facecolor=WHITE)
names=['X','p25','F / P25F','A5 / D','DF'];values=[.958,.958,.957,.956,.955]
for i,(n,v) in enumerate(zip(names,values)):
    color=TEAL if i<2 else BLUE
    plot.scatter(v,i,s=180,color=color,zorder=3)
    plot.text(v+.00028,i,f'{v:.3f}',va='center',fontsize=13,color=INK,weight='bold')
plot.set_yticks(range(5),names);plot.invert_yaxis();plot.set_xlim(.95,.96)
plot.set_xticks([.950,.952,.954,.956,.958,.960]);plot.tick_params(axis='both',labelsize=12,length=0,pad=10)
plot.grid(axis='x',color=LINE,lw=1);plot.set_axisbelow(True)
for spine in plot.spines.values(): spine.set_visible(False)
plot.set_xlabel('Public AUC display · dot plot, axis 0.950–0.960',fontsize=10,labelpad=18,color=MUTED)
text(ax,1020,228,'UNRECORDED AT CUTOFF',13,ORANGE,'bold')
for y,name,detail in [(288,'U','Accepted · outcome/score not recorded'),(372,'P4','Accepted · outcome/score not recorded'),(456,'U4','Accepted · outcome/score not recorded'),(540,'F4','Preparation · no accepted ref recorded')]:
    text(ax,1020,y,name,20,INK,'bold');text(ax,1110,y+4,detail,10,MUTED)
text(ax,90,704,'A tied rounded public display does not prove equal exact scores or private performance.',13,INK,'bold')
footer(ax,762,'Cutoff: 2026-10-22 18:15 KST. Sources: D-B-260/274/276/280/281 and immutable acceptance ledger.','No leaderboard query was made. Pending candidates are not plotted as zero; final pair/manual confirmation remains unrecorded.')
save(fig,'public-score-snapshot.png','Five recorded public displays and separate unrecorded candidate states.',['D-B-260','D-B-274','D-B-276','D-B-280','D-B-281','submit_ledger.jsonl'])

fig,ax=canvas('Full-bag training: replacement vs addition','같은 모델도 결합 방식에 따라 달랐다 · local holdout evidence, not a private-score forecast',9)
box(ax,60,205,1480,473)
text(ax,92,232,'Mean change across four local evaluation rulers',16,INK,'bold')
text(ax,92,269,'Two distinct, fixed comparisons — do not treat them as interchangeable experiments.',12,MUTED)
plot=fig.add_axes([.265,.40,.59,.25],facecolor=WHITE)
vals=[-.0045,.001449]
labels=['Replace seed0\nseed0 교체','Add fourth member\n네 번째 멤버 추가']
for i,v in enumerate(vals):
    col=ORANGE if v<0 else TEAL
    plot.barh(i,v,height=.33,color=col,alpha=.85)
    plot.text(v+(.00016 if v>0 else -.00016),i,('~ -0.0045' if v<0 else '+0.001449'),ha='left' if v>0 else 'right',va='center',size=14,color=col,weight='bold')
plot.axvline(0,color=MUTED,lw=1.4);plot.set_xlim(-.0065,.003);plot.set_ylim(1.6,-.6)
plot.set_yticks(range(2),labels);plot.set_xticks([-.006,-.004,-.002,0,.002]);plot.tick_params(length=0,pad=12)
plot.grid(axis='x',color=LINE);plot.set_axisbelow(True)
for spine in plot.spines.values():spine.set_visible(False)
plot.set_xlabel('Change in mean macro AUC / 평균 macro AUC 변화',fontsize=11,labelpad=16,color=MUTED)
text(ax,94,641,'Decision: test K4 as an additional member → P4 / U4; keep the original trio.',14,INK,'bold')
text(ax,94,704,'The exposed/report-derived rulers share limitations. Their gain does not establish hidden/private gain.',12,MUTED)
footer(ax,762,'Sources: D-A-052 / D-B-264 (replacement); D-B-278 (addition, independent result replay).','Replacement mean is rounded. Addition improved all four rulers (+0.00085 to +0.00201); registered cross-check status is unchanged.')
save(fig,'fullbag-local-evidence.png','Local full-bag replacement and addition contrasts with their distinct controls.',['D-A-052','D-B-264','D-B-278'])

(OUT/'asset-manifest.json').write_text(json.dumps({'version':'1.0.0','cutoff_kst':'2026-10-22 18:15','generator':'tools/render_research_explainer_20261022.py','method':'Original deterministic Matplotlib diagrams and aggregate-data charts; no image-generation API or patient imagery.','assets':manifest},indent=2),encoding='utf-8')
print(json.dumps({'assets':len(manifest),'total_bytes':sum(x['bytes'] for x in manifest),'output':str(OUT)}))
