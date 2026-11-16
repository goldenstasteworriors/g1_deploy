"""Diagnose saved predictions without inference, actuator I/O, or wait ticks.

Recorded observation trajectories are smoothness references, not prediction
accuracy labels: the saved policy used achieved simulated state as feedback.
"""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab_low.skills import ARM_INDICES


def metrics(q, hz):
    q = np.asarray(q, dtype=float)
    d = np.diff(q, axis=-2)
    a = np.diff(q, n=2, axis=-2)*hz**2
    j = np.diff(q, n=3, axis=-2)*hz**3
    return dict(max_step_rad=float(np.abs(d).max()),
        p95_step_rad=float(np.percentile(np.abs(d),95)),
        acceleration_rms_rad_s2=float(np.sqrt(np.mean(a*a))),
        jerk_rms_rad_s3=float(np.sqrt(np.mean(j*j))),
        reversal_fraction=float(np.mean(d[...,1:,:]*d[...,:-1,:]<0)))


def blend(raw, starts, initial, window, velocity_match=False):
    """Fixed-prediction ablation. Gripper targets are never modified."""
    sent = raw.copy()
    for k,b in enumerate(starts):
        end = starts[k+1] if k+1<len(starts) else len(raw)
        previous = initial if b==0 else sent[b-1]
        if velocity_match:
            v_previous = np.zeros(16) if b<2 else (sent[b-1]-sent[b-2])*30
            v_raw = (raw[min(b+1,end-1)]-raw[b])*30
            q_raw_zero = raw[b]-v_raw/30
            offset = previous-q_raw_zero
        else:
            offset = previous-raw[b]
        for n in range(min(window,end-b)):
            if velocity_match:
                u = (n+1)/window
                h0 = 1-10*u**3+15*u**4-6*u**5
                h1 = u-6*u**3+8*u**4-3*u**5
                correction = h0*offset+(window/30)*h1*(v_previous-v_raw)
            else:
                u = n/max(1,window-1)
                correction = (1-10*u**3+15*u**4-6*u**5)*offset
            sent[b+n,8:15] += correction[8:15]
    np.testing.assert_array_equal(sent[:,[7,15]],raw[:,[7,15]])
    return sent


def seams(q, starts, initial):
    rows=[]
    for b in starts[1:]:
        if b+1>=len(q):continue
        before=(q[b-1,8:15]-q[b-2,8:15])*30
        crossing=(q[b,8:15]-q[b-1,8:15])*30
        after=(q[b+1,8:15]-q[b,8:15])*30
        denom=np.linalg.norm(before)*np.linalg.norm(after)
        rows.append(dict(step=int(b),max_position_jump_rad=float(np.abs(q[b,8:15]-q[b-1,8:15]).max()),
            max_velocity_jump_rad_s=float(np.abs(crossing-before).max()),
            incoming_speed_rad_s=float(np.linalg.norm(before)),
            crossing_speed_rad_s=float(np.linalg.norm(crossing)),
            outgoing_speed_rad_s=float(np.linalg.norm(after)),
            velocity_direction_cosine=None if denom<1e-12 else float(before@after/denom)))
    return dict(rows=rows,max_position_jump_rad=max(r['max_position_jump_rad'] for r in rows),
        mean_position_jump_rad=float(np.mean([r['max_position_jump_rad'] for r in rows])),
        mean_velocity_jump_rad_s=float(np.mean([r['max_velocity_jump_rad_s'] for r in rows])),
        forced_zero_crossings=sum(r['crossing_speed_rad_s']<1e-10 and r['incoming_speed_rad_s']>1e-6 for r in rows))


def native_50hz(raw,initial):
    """Call the actual formal ArmTargetBuffer with a fake model, no transport."""
    from lab_low.g1_bridge import ArmTargetBuffer
    class Model:
        num_dofs=29
        def get_body_actuated_joint_indices(self):return list(range(29))
        def get_joint_group_indices(self,name):
            assert name=='upper_body'
            return list(range(15,29))
    state=np.zeros(31);state[ARM_INDICES]=initial
    buffer=ArmTargetBuffer(Model(),max_arm_speed=1000.)
    buffer.observe(dict(q=state[:29],left_hand_dex1_q=state[29:30],right_hand_dex1_q=state[30:31]))
    times=np.arange(int(np.ceil(len(raw)/30*50))+1)/50
    values=[];last=-1
    for t in times:
        i=min(int(np.floor(t*30+1e-8)),len(raw)-1)
        if i!=last:
            buffer.receive(dict(lab_target=raw[i].tolist(),lab_binding_id='offline-diagnostic',
                lab_sequence=i,lab_reset=last<0,lab_skill='pick_up_test_tube',lab_hand='right'),now=float(t))
            last=i
        values.append(buffer.sample(now=float(t))['target_upper_body_pose'][7:14])
    knots=np.arange(len(raw)+1)/30
    q=np.vstack([initial[8:15],raw[:,8:15]])
    correct=np.stack([np.interp(times,knots,q[:,j]) for j in range(7)],axis=1)
    return times,np.asarray(values),correct


def fk_point(arms,initial_body):
    """Arm-only FK at a nominal point 10cm along wrist local X, fixed base."""
    from decoupled_wbc.control.robot_model.instantiation.g1 import instantiate_g1_robot_model
    from lab_low.gravity import ArmGravityCompensator
    gravity=ArmGravityCompensator(instantiate_g1_robot_model(with_hands=False))
    pin,model,data=gravity.pin,gravity.model,gravity.data
    jid=model.getJointId('right_wrist_yaw_joint')
    values=[]
    for arm in np.asarray(arms).reshape(-1,7):
        body=initial_body.copy();body[22:29]=arm
        q=np.zeros(model.nq);q[gravity.q_indices]=body
        pin.forwardKinematics(model,data,q)
        pose=data.oMi[jid]
        values.append(pose.translation+pose.rotation@np.array([.10,0,0]))
    return np.asarray(values).reshape(np.asarray(arms).shape[:-1]+(3,))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--rollout',type=Path,required=True)
    ap.add_argument('--dataset',type=Path,required=True)
    ap.add_argument('--episode',type=int,default=0)
    ap.add_argument('--sonic-root',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    sys.path.insert(0,str(args.sonic_root))
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    info=json.loads((args.dataset/'meta/info.json').read_text())
    path=args.dataset/info['data_path'].format(episode_index=args.episode,episode_chunk=args.episode//info['chunks_size'])
    gt=np.asarray(pq.read_table(path,columns=['observation.state'])['observation.state'].to_pylist())
    names=info['features']['observation.state']['names'][22:29]
    summary=json.loads((args.rollout/'summary.json').read_text())
    chunks=json.loads((args.rollout/'predicted_chunks.json').read_text())
    adopted=summary['async_stats']['adoptions']
    matched=[r for r in chunks if r['frame']+33<=len(gt)]
    model=np.asarray([r['actions'] for r in matched])[:,:,8:15]
    recorded=np.asarray([gt[r['frame']+1:r['frame']+33,22:29] for r in matched])
    n=summary['final_frame']-summary['start_frame']
    raw=np.zeros((n,16));sg_raw=np.zeros((n,16));starts=[]
    for r in adopted:
        b=r['adoption_step']
        if b>=n:break
        starts.append(b)
        row=next(c for c in chunks if c['frame']==summary['start_frame']+r['input_step'])
        count=min(r['available'],n-b);skip=r['skip_steps']
        full=np.asarray(row['actions'])
        raw[b:b+count]=full[skip:skip+count]
        filtered=full.copy()
        filtered[:,8:15]=savgol_filter(full[:,8:15],5,2,axis=0)
        sg_raw[b:b+count]=filtered[skip:skip+count]
    saved=np.load(args.rollout/'trajectory.npz')
    initial=saved['sim_state'][0,ARM_INDICES]
    variants={'raw':raw,'offset10':blend(raw,starts,initial,10),
              'velocity_matched6_candidate':blend(raw,starts,initial,6,True),
              'intra_sg5_velocity6_candidate':blend(sg_raw,starts,initial,6,True)}
    # Check fixed replay alignment against targets actually saved by the run.
    if summary.get('chunk_smoothing_steps')==10:
        executed=[]
        for r in adopted:
            b=r['adoption_step']
            if b>=n:break
            c=r.get('adoption_clock_step',b)
            executed.extend(saved['targets'][c:c+min(r['available'],n-b),8:15])
        np.testing.assert_allclose(variants['offset10'][:,8:15],executed,atol=1e-10,rtol=0)
    intra=dict(model=metrics(model,30),recorded_reference=metrics(recorded,30),matched_chunks=len(matched),
               savgol5_candidate=metrics(savgol_filter(model,5,2,axis=1),30),
               per_joint=[dict(joint=name,model=metrics(model[:,:,j:j+1],30),
                               recorded=metrics(recorded[:,:,j:j+1],30)) for j,name in enumerate(names)])
    result=dict(scope='saved fixed predictions; all inference/hold ticks excluded; no new model or robot calls',
        source=str(args.rollout),intra=intra,
        stitched={k:dict(trajectory=metrics(v[:,8:15],30),boundaries=seams(v,starts,initial)) for k,v in variants.items()})
    control_input=variants['offset10']
    control_t,native,linear=native_50hz(control_input,initial)
    result['control_50hz']=dict(input_trajectory='same offset10-smoothed targets for both samplers',
        native_current_bridge=metrics(native,50),time_correct_linear=metrics(linear,50),
        native_all_joint_hold_fraction=float(np.mean(np.max(np.abs(np.diff(native,axis=0)),axis=1)<1e-10)),
        linear_all_joint_hold_fraction=float(np.mean(np.max(np.abs(np.diff(linear,axis=0)),axis=1)<1e-10)))
    initial_body=saved['sim_state'][0,:29]
    # Batch once: the reference fixes all other joints identically.
    all_arms=np.concatenate([model.reshape(-1,7),recorded.reshape(-1,7)] +
                           [v[:,8:15] for v in variants.values()])
    all_xyz=fk_point(all_arms,initial_body)
    size=model.shape[0]*32
    model_xyz=all_xyz[:size].reshape(model.shape[0],32,3)
    gt_xyz=all_xyz[size:2*size].reshape(model.shape[0],32,3)
    result['nominal_gripper_point_fk']=dict(point='10cm along right wrist-yaw joint local X; uncalibrated proxy TCP',
        fixed_other_joints=True,model_intra_acceleration_rms_m_s2=float(np.sqrt(np.mean(np.diff(model_xyz,n=2,axis=1)**2))*900),
        reference_intra_acceleration_rms_m_s2=float(np.sqrt(np.mean(np.diff(gt_xyz,n=2,axis=1)**2))*900),
        stitched={})
    for k,key in enumerate(variants):
        xyz=all_xyz[2*size+k*n:2*size+(k+1)*n]
        b=np.asarray(starts[1:])
        result['nominal_gripper_point_fk']['stitched'][key]=dict(
            max_boundary_displacement_mm=float(np.linalg.norm(xyz[b]-xyz[b-1],axis=1).max()*1000),
            acceleration_rms_m_s2=float(np.sqrt(np.mean(np.diff(xyz,n=2,axis=0)**2))*900))
    result['limits']=['Recorded state is a smoothness reference; it is not matching policy feedback ground truth.',
        'Velocity-matched curve is an offline candidate, not deployed or a proven task-success improvement.',
        'Savitzky-Golay window5 polynomial2 is applied to each complete predicted chunk before trimming, active arm only.',
        'Command-space metrics are not actual motor acceleration or jerk.',
        'The 50Hz comparison calls the real ArmTargetBuffer using the same saved raw targets; no ROS/DDS.',
        'No physical replay is inferred from trajectory resampling.']
    (out/'metrics.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    np.savez_compressed(out/'fixed_predictions.npz',**variants,starts=starts,control_time=control_t,
        native_50hz=native,time_correct_50hz=linear)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    # Use the most oscillatory complete prediction window, chosen by acceleration.
    ci=int(np.argmax(np.mean(np.diff(model,n=2,axis=1)**2,axis=(1,2))))
    fig,axes=plt.subplots(4,2,figsize=(13,10),layout='constrained');t=np.arange(1,33)/30
    for j,ax in enumerate(axes.flat[:7]):
        ax.plot(t,model[ci,:,j]-model[ci,0,j],'.-',label='Raw prediction',color='#2376ad')
        ax.plot(t,recorded[ci,:,j]-recorded[ci,0,j],label='Recorded reference',color='#333333')
        sg=savgol_filter(model[ci,:,j],5,2)
        ax.plot(t,sg-sg[0],label='Intra-chunk SG5 candidate',color='#be7626',alpha=.8)
        ax.set(title=names[j].replace('right_',''),xlabel='Prediction time (s)',ylabel='Relative angle (rad)');ax.grid(alpha=.2)
    axes.flat[0].legend();axes.flat[7].axis('off')
    axes.flat[7].text(0,.6,'Angles are shifted to start at zero.\nReference is a smoothness comparison,\nnot a prediction-accuracy label.\nAll 32 predicted steps are shown.',va='center')
    fig.suptitle(f'Inside one chunk: observation dataset frame {matched[ci]["frame"]}, 10 denoise steps')
    fig.savefig(out/'intra_chunk.png',dpi=160);plt.close(fig)
    worst=max(result['stitched']['raw']['boundaries']['rows'],key=lambda r:r['max_position_jump_rad'])['step']
    joint=int(np.argmax(np.abs(raw[worst,8:15]-raw[worst-1,8:15])))
    t=np.arange(1,n+1)/30;fig,axes=plt.subplots(3,1,figsize=(13,9),sharex=True,layout='constrained')
    for key,color in [('raw','#bd3d38'),('offset10','#2376ad'),('velocity_matched6_candidate','#29834f'),
                      ('intra_sg5_velocity6_candidate','#9b50a1')]:
        q=variants[key][:,8+joint];v=np.diff(q)*30;a=np.diff(q,n=2)*900
        axes[0].plot(t,q,'.-',ms=3,label=key,color=color);axes[1].plot(t[1:],v,'.-',ms=3,color=color);axes[2].plot(t[2:],a,'.-',ms=3,color=color)
    for ax in axes:
        ax.axvline(t[worst],color='black',ls='--',alpha=.6);ax.grid(alpha=.2);ax.set_xlim(max(0,t[worst]-.25),t[worst]+.55)
    axes[0].legend();axes[0].set_ylabel('Target angle (rad)');axes[1].set_ylabel('Discrete velocity (rad/s)');axes[2].set_ylabel('Discrete acceleration (rad/s2)');axes[2].set_xlabel('Execution time, excluding waits (s)')
    fig.suptitle('Same fixed predictions at the worst raw chunk boundary: '+names[joint])
    fig.savefig(out/'chunk_boundary.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(13,7),sharex=True,layout='constrained')
    axes[0].plot(control_t,native[:,joint],'.-',label='Current ArmTargetBuffer, 50Hz')
    axes[0].plot(control_t,linear[:,joint],'.-',label='Timestamp-correct interpolation, 50Hz')
    axes[0].plot(t,control_input[:,8+joint],'kx',label='30Hz smoothed target at its trained time')
    axes[1].plot(control_t[1:],np.diff(native[:,joint])*50,'.-',label='Current bridge')
    axes[1].plot(control_t[1:],np.diff(linear[:,joint])*50,'.-',label='Time-correct interpolation')
    left=max(0,t[worst]-.4);right=t[worst]-.05
    for ax in axes:ax.set_xlim(left,right);ax.grid(alpha=.2);ax.legend()
    axes[0].set_ylabel('Target angle (rad)');axes[1].set_ylabel('Command velocity (rad/s)');axes[1].set_xlabel('Execution time (s)')
    fig.suptitle('Frequency conversion with identical predictions; inference waits excluded')
    fig.savefig(out/'control_timing.png',dpi=160);plt.close(fig)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
