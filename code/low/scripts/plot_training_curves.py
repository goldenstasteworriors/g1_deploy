"""Plot auditable train/validation curves from a captured JSONL snapshot."""
import argparse
import csv
import json
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--snapshot',type=Path,required=True)
    args=ap.parse_args()
    out=args.snapshot.resolve().parent
    cache=out/'.matplotlib';cache.mkdir(exist_ok=True)
    os.environ['MPLCONFIGDIR']=str(cache)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager,ticker
    import numpy as np
    font=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=str(font)).get_name(),
        'axes.unicode_minus':False,'font.size':11,'axes.spines.top':False,'axes.spines.right':False,
        'axes.edgecolor':'#9ca3af','axes.labelcolor':'#374151','xtick.color':'#4b5563','ytick.color':'#4b5563'})
    data=json.loads(args.snapshot.read_text())
    titles={'return_dropper':'放回滴管','water_bath':'放入水浴',
            'take_tube_light':'取试管 · 光照随机化','return_tube':'试管归架'}
    summary={}
    for name,row in data['models'].items():
        for key,fields in [('train',['step','loss_action','loss_video','loss_total','grad_norm']),
                           ('validation',['step','action','video','total'])]:
            with (out/(name+'_'+key+'.csv')).open('w') as f:
                writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
                writer.writeheader();writer.writerows(row[key])
        vals=row['validation']
        best=min(vals,key=lambda x:x['action'])
        summary[name]=dict(step=row['train'][-1]['step'],first_action_validation=vals[0]['action'],
            last_action_validation=vals[-1]['action'],best_action_validation=best['action'],best_step=best['step'])
    for metric,title in [('action','动作损失'),('video','视觉预测损失')]:
        fig,axes=plt.subplots(2,2,figsize=(14,9),layout='constrained')
        fig.set_facecolor('#ffffff')
        stamp=data['captured_at'][:19].replace('T',' ')
        fig.suptitle(f'{title}｜四个单卡实验\n{stamp}（北京时间） · 目标 30,000 步',fontsize=18,fontweight='bold')
        handles=None
        for ax,(name,row) in zip(axes.flat,data['models'].items()):
            steps=np.array([x['step'] for x in row['train']])
            y=np.array([x['loss_'+metric] for x in row['train']])
            vs=np.array([x['step'] for x in row['validation']])
            vy=np.array([x[metric] for x in row['validation']])
            assert np.isfinite(y).all() and np.isfinite(vy).all() and (y>=0).all() and (vy>0).all()
            assert np.all(np.diff(steps)>0) and np.all(np.diff(vs)>0)
            # Trailing arithmetic mean of up to 200 actual optimizer steps.
            c=np.r_[0.,np.cumsum(y)];end=np.arange(1,len(y)+1);start=np.maximum(0,end-200)
            smooth=(c[end]-c[start])/(end-start)
            a,=ax.plot(steps,np.ma.masked_less_equal(y,0),color='#adb5bd',alpha=.25,lw=.45,label='训练原值',rasterized=True)
            b,=ax.plot(steps,np.ma.masked_less_equal(smooth,0),color='#2563eb',lw=1.8,label='训练均值（最近 200 步）')
            v,=ax.plot(vs,vy,color='#e76f31',lw=1.8,marker='o',ms=4,label='验证集（每 1000 步）')
            best=int(vy.argmin());ax.scatter([vs[best]],[vy[best]],s=115,marker='*',color='#b45309',zorder=5)
            handles=[a,b,v]
            ax.set_title(f'GPU {row["gpu"]}  {titles[name]}   {steps[-1]:,} / 30,000',loc='left',fontsize=12,pad=12)
            ax.text(.98,.96,f'最新验证：{vy[-1]:.4g}\n本图最低：{vy[best]:.4g}（{vs[best]:,} 步）',
                    ha='right',va='top',transform=ax.transAxes,fontsize=10,
                    bbox=dict(boxstyle='round,pad=.35',fc='white',ec='#e5e7eb',alpha=.95))
            ax.set_yscale('log');ax.set_xlim(0,30000)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(5000))
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x,pos:f'{int(x/1000)}k'))
            ax.grid(True,which='major',color='#e5e7eb',lw=.7);ax.set_axisbelow(True)
            ax.set_xlabel('训练步数');ax.set_ylabel('Loss（对数坐标）')
        fig.legend(handles=handles,loc='outside lower center',ncol=3,frameon=False,fontsize=11,
                   title='星标：该图最低验证损失；不是操作成功率。不同实验的归一化统计不同。\n零值原始点不画在对数轴，平滑计算和 CSV 仍保留。')
        fig.savefig(out/(metric+'_loss.png'),dpi=170,facecolor='white')
        fig.savefig(out/(metric+'_loss.svg'),facecolor='white')
        plt.close(fig)
    (out/'summary.json').write_text(json.dumps(dict(captured_at=data['captured_at'],models=summary),indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
