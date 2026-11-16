"""Fixed-base G1/Dex1 physics feedback diagnostic, with recorded RGB input.

No calibrated objects, liquid dynamics, grasp attachment or task-success labels.
Only the first state initializes simulation. All subsequent policy states are
achieved MuJoCo joint positions; recorded states are only used for comparison.
"""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
import argparse
import copy
import json
from pathlib import Path
import shlex
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET

import cv2
import mujoco
import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab_low.skills import ARM_INDICES, action_mask

REMOTE = '/share/personal/yekangjie/lab-low/code'
ENV = '/share/personal/yekangjie/lab-low/envs/openwam'
DEFAULT_ASSET = '/home/ykj/project/WM/openwam_deployment/test_tube_20261027/assets/unitree_ros/robots/g1_description'
CHECKPOINTS = {
    'general': REMOTE+'/outputs/boxed_v3_30k/general/2026-11-10_03-21-17/checkpoint_step_30000.safetensors',
    'take_tube': REMOTE+'/outputs/boxed_v3_30k/take_tube/2026-11-10_14-30-10/checkpoint_step_30000.safetensors',
}


def slider(q):
    return -.020 + .0445*np.clip(q, 0, 5.4)/5.4


def motor(q):
    return np.clip((np.asarray(q)+.020)*5.4/.0445, 0, 5.4)


class Physics:
    def __init__(self, out, asset, names, initial, active, deployment_control=False):
        self.names = names
        tree = ET.parse(asset/'g1_29dof_mode_15_with_dex1_1.urdf')
        self.gravity = None
        self.active = active
        self.held = initial[ARM_INDICES].copy()
        if deployment_control:
            import decoupled_wbc
            from lab_low.gravity import ArmGravityCompensator, XR_KP, XR_KD
            from decoupled_wbc.control.robot_model.instantiation.g1 import instantiate_g1_robot_model
            self.robot = instantiate_g1_robot_model(with_hands=False)
            self.gravity = ArmGravityCompensator(self.robot)
            dex = tree.getroot()
            tree = ET.parse(Path(decoupled_wbc.__file__).parent/'control/robot_model/model_data/g1/g1_29dof.urdf')
            root = tree.getroot()
            for side in ('left','right'):
                root.remove(root.find(f"./link[@name='{side}_rubber_hand']"))
                root.remove(root.find(f"./joint[@name='{side}_hand_palm_joint']"))
                for node in dex:
                    name = node.get('name','')
                    if name.startswith(side+'_dex1_') or name == side+'_base_joint':
                        node = copy.deepcopy(node)
                        inert = node.find('inertial')
                        if inert is not None:
                            factor = .55 / .364946
                            mass = inert.find('mass')
                            mass.set('value', str(float(mass.get('value'))*factor))
                            inertia = inert.find('inertia')
                            for k,v in list(inertia.attrib.items()):
                                inertia.set(k,str(float(v)*factor))
                        root.append(node)
        tree.find('./mujoco/compiler').set('meshdir', str(asset/'meshes'))
        for mesh in tree.findall('.//mesh'):
            mesh.set('filename', Path(mesh.get('filename')).name)
        tree.write(out/'portable.urdf')
        model = mujoco.MjModel.from_xml_path(str(out/'portable.urdf'))
        mujoco.mj_saveLastXML(str(out/'raw.xml'), model)
        initial_joints = {names[i]: float(initial[i]) for i in range(29)}
        controlled = []
        for side, arm_start, hand_index in [('left',15,29), ('right',22,30)]:
            controlled += names[arm_start:arm_start+7]
            for k in (1,2):
                name = f'{side}_dex1_finger_joint_{k}'
                initial_joints[name] = float(slider(initial[hand_index]))
                controlled.append(name)
        tree = ET.parse(out/'raw.xml')
        xml = tree.getroot()
        world = xml.find('worldbody')
        anchor = ET.Element('body', name='fixed_pelvis_anchor', pos='0 0 0.80')
        for element in list(world):
            world.remove(element)
            anchor.append(element)
        world.append(anchor)
        ET.SubElement(world,'geom',name='floor',type='plane',size='3 3 .1',rgba='.22 .25 .29 1',group='2')
        ET.SubElement(world,'light',pos='1 -2 3',dir='-1 2 -3',diffuse='.8 .8 .8')
        option = xml.find('option')
        if option is None:
            option = ET.SubElement(xml,'option')
        option.set('timestep',str(1/300))
        option.set('integrator','implicitfast')
        visual = ET.SubElement(xml,'visual')
        ET.SubElement(visual,'global',offwidth='960',offheight='720')
        eq = ET.SubElement(xml,'equality')
        act = ET.SubElement(xml,'actuator')
        expanded_active = np.r_[active[:7], active[7], active[7], active[8:15], active[15], active[15]]
        moving = set(controlled) if deployment_control else {n for n,a in zip(controlled,expanded_active) if a}
        for joint in xml.findall('.//joint'):
            name = joint.get('name')
            if name not in initial_joints:
                continue
            joint.set('damping','0' if deployment_control and 'dex1' not in name else '1')
            joint.set('armature','.001' if 'dex1' in name else '.02')
            if name not in moving:
                ET.SubElement(eq,'joint',joint1=name,polycoef=f'{initial_joints[name]} 0 0 0 0')
        for i,name in enumerate(controlled):
            joint = next(j for j in xml.findall('.//joint') if j.get('name') == name)
            arm_i = i if i < 7 else i-2
            kp = 1000 if 'dex1' in name else XR_KP[arm_i] if deployment_control else 80
            kd = 10 if 'dex1' in name else XR_KD[arm_i] if deployment_control else 8
            ET.SubElement(act,'position',name=name,joint=name,kp=str(kp),
                kv=str(kd),ctrlrange=joint.get('range'),ctrllimited='true')
        tree.write(out/'scene.xml')
        self.model = mujoco.MjModel.from_xml_path(str(out/'scene.xml'))
        self.data = mujoco.MjData(self.model)
        for name, value in initial_joints.items():
            self.data.qpos[self.model.joint(name).qposadr[0]] = value
        self.controlled = controlled
        self.qi = np.array([self.model.joint(n).qposadr[0] for n in controlled])
        self.vi = np.array([self.model.joint(n).dofadr[0] for n in controlled])
        self.ai = np.array([self.model.actuator(n).id for n in controlled])
        self.arm_vi = self.vi[[*range(7), *range(9,16)]]
        mujoco.mj_forward(self.model,self.data)
        self.data.ctrl[self.ai] = self.data.qpos[self.qi]

    def state(self):
        q = [self.data.qpos[self.model.joint(n).qposadr[0]] for n in self.names[:29]]
        q += [float(motor(self.data.qpos[self.qi[7:9]].mean())),
              float(motor(self.data.qpos[self.qi[16:18]].mean()))]
        return np.asarray(q, np.float32)

    def step(self, target):
        if self.gravity is not None:
            target = np.where(self.active, target, self.held)
            self.held = target.copy()
        control = np.r_[target[:7], slider(target[7]), slider(target[7]),
                        target[8:15], slider(target[15]), slider(target[15])]
        bounded = np.clip(control,self.model.actuator_ctrlrange[self.ai,0],self.model.actuator_ctrlrange[self.ai,1])
        self.data.ctrl[self.ai] = bounded
        if self.gravity is not None:
            state = self.state()
            obs = dict(q=self.robot.initial_body_pose.copy(), floating_base_pose=np.array([0,0,.8,1,0,0,0]))
            obs['q'][self.gravity.body] = state[:29]
            command = state[:29].copy()
            command[15:22],command[22:29] = bounded[:7],bounded[9:16]
            tau = self.gravity.sample_target(obs, command)[15:29]
        for _ in range(10):
            self.data.qfrc_applied[:] = 0
            self.data.qfrc_applied[self.arm_vi] = self.data.qfrc_bias[self.arm_vi] if self.gravity is None else tau
            mujoco.mj_step(self.model,self.data)
        if not np.isfinite(self.data.qpos).all():
            raise RuntimeError('Nonfinite simulated state')
        return int(np.any(control != bounded) or target[7] < 0 or target[7] > 5.4 or target[15] < 0 or target[15] > 5.4)


class Worker:
    def __init__(self, variant, gpu, log):
        self.log = log.open('w')
        command = (f'cd {REMOTE} && CUDA_VISIBLE_DEVICES={gpu} PYTHONNOUSERSITE=1 PYTHONPATH={REMOTE} '
                   f'TMPDIR={REMOTE}/.cache/tmp XDG_CACHE_HOME={REMOTE}/.cache HF_HUB_OFFLINE=1 '
                   f'TRITON_CACHE_DIR={REMOTE}/.cache/triton TORCH_EXTENSIONS_DIR={REMOTE}/.cache/torch_extensions '
                   f'TORCHINDUCTOR_CACHE_DIR={REMOTE}/.cache/torchinductor CUDA_CACHE_PATH={REMOTE}/.cache/cuda '
                   f'TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 '
                   f'{ENV}/bin/python -u scripts/sim_inference_worker.py '
                   f'--checkpoint {CHECKPOINTS[variant]} --manifest data/boxed_v3_suite/{variant}.json')
        self.process = subprocess.Popen(['ssh','-o','ConnectTimeout=180','-o','ConnectionAttempts=5',
            '-o','ServerAliveInterval=30','-o','ServerAliveCountMax=10','--','A800-1', command],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1)
        result = self.read()
        if not result.get('ready'):
            raise RuntimeError(result)

    def read(self):
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f'Model worker exited; see {self.log.name}')
        return json.loads(line)

    def request(self, payload):
        self.process.stdin.write(json.dumps(payload)+'\n')
        self.process.stdin.flush()
        return self.read()

    def close(self):
        self.process.stdin.close()
        try:
            self.process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=30)
        self.log.close()


class DeploymentWorker:
    """Remote dataset decoding + existing policy HTTP; never imports robot I/O."""
    def __init__(self, args, log):
        self.rpc_lock = threading.Lock()
        self.log = log.open('w')
        code = r'''
import sys,json,uuid,urllib.request,time
from pathlib import Path
root=Path(sys.argv[1]); variant=sys.argv[2]
sys.path[:0]=[str(root/'code/low'),str(root/'code/low/scripts')]
from lab_low.recorded_vision import RecordedDataset,RecordedEpisode
from run_g1_lab_client import rpc as raw_rpc
def rpc(url, packet):
 return raw_rpc(url, dict(packet, client_scope='simulation'))
url='http://127.0.0.1:8849'
health=json.load(urllib.request.build_opener(urllib.request.ProxyHandler({})).open(url+'/',timeout=10))
assert health.get('ready') and '/models/'+variant+'/' in health['checkpoint'],health
assert 'isolated_client_scopes_v1' in health.get('protocol_features',[]), 'Restart policy with isolated simulation sessions before evaluating'
print(json.dumps(health),flush=True)
source=None;binding='simulation-'+str(uuid.uuid4());index=0;batch_seconds=0.
try:
 for line in sys.stdin:
  request=json.loads(line);began=time.monotonic()
  if request['op']=='stop':
   result=rpc(url,dict(op='stop',binding_id=binding));source.close();source=None
   print(json.dumps(result),flush=True);continue
  if request['op']=='reset':
   if source: source.close()
   data=RecordedDataset(root/'assets/datasets/lab_primitives_train_20261110_boxed_v3',health['supported_skill_hands'])
   source=RecordedEpisode(data,request['episode'],request['start_frame']);index=0;batch_seconds=0.
   result=rpc(url,source.reset_packet(request['state'],binding))
  else:
   observations=[];preprocess_seconds=0.
   for packet in request['observations']:
    if packet['frame'] == source.frame+1: source.advance()
    assert source.frame==packet['frame']
    index+=1
    encoded_at=time.monotonic()
    observations.append(dict(frame_index=index,state=packet['state'],images=source.encode(source.current_images,'.jpg')))
    preprocess_seconds+=time.monotonic()-encoded_at
   if observations:
    observed_at=time.monotonic()
    rpc(url,dict(op='observe_batch',binding_id=binding,observations=observations))
    preprocess_seconds+=time.monotonic()-observed_at
   if request['op']=='observe_only':
    # Client RGB is decoded on each control tick already; this second decoding
    # exists only to avoid shipping images over SSH. Do not charge it twice.
    batch_seconds=preprocess_seconds;result=dict(ok=True,observed=len(observations))
   else:
    result=rpc(url,dict(op='predict',binding_id=binding,denoise_steps=request.get('denoise_steps',20)))
    result['pipeline_seconds']=batch_seconds+time.monotonic()-began;batch_seconds=0.
  result['frame']=source.frame
  print(json.dumps(result),flush=True)
finally:
 if source:
  source.close();rpc(url,dict(op='stop',binding_id=binding))
'''
        command = ' '.join(map(shlex.quote, [args.rpc_root+'/.conda_controller/bin/python','-u','-c',code,args.rpc_root,args.variant]))
        self.process = subprocess.Popen(['ssh','-o','ConnectTimeout=40','-o','ServerAliveInterval=10',
            '--',args.rpc_host,command],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,bufsize=1)
        health = self.read()
        if not health.get('ready'):
            raise RuntimeError(health)
        self.checkpoint = health['checkpoint']

    read = Worker.read
    def request(self, payload):
        # stop may arrive while the inference thread is reading a prediction.
        with self.rpc_lock:
            return Worker.request(self, payload)
    close = Worker.close


def deployment_rollout(args, worker, out, episode):
    """Current deployment protocol; only initial recorded state enters physics."""
    from lab_low.recorded_vision import RecordedDataset, RecordedEpisode
    from run_g1_recorded_client import run_session_async
    out.mkdir(parents=True,exist_ok=False)
    dataset = RecordedDataset(args.dataset, [('pick_up_test_tube','right')])
    source = RecordedEpisode(dataset,episode,args.start_frame)
    fields = dict(episode_index=episode,episode_chunk=episode//dataset.info['chunks_size'])
    gt = np.asarray(pq.read_table(dataset.root/dataset.info['data_path'].format(**fields),
        columns=['observation.state'])['observation.state'].to_pylist(),np.float32)
    names = dataset.info['features']['observation.state']['names']
    active = action_mask(source.task['skill'],source.task['hand'])
    sim = Physics(out,Path(args.asset),names,gt[args.start_frame],active,deployment_control=True)
    renderer = mujoco.Renderer(sim.model,height=576,width=768)
    camera = mujoco.MjvCamera();camera.lookat[:]=[.15,0,1.13]
    camera.distance,camera.azimuth,camera.elevation=1.55,135,-13
    view = mujoco.MjvOption();view.geomgroup[3:]=0
    encoder = subprocess.Popen([args.ffmpeg,'-hide_banner','-loglevel','error','-y',
        '-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1280x960','-r','30','-i','-',
        '-an','-c:v','libx264','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'simulation.mp4')],stdin=subprocess.PIPE)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
    titlefont=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',22)
    from lab_low.dataset import CAMERAS
    reference=Image.fromarray(source.reference_images[CAMERAS[0]])
    overlay=ImageDraw.Draw(reference)
    for mark in source.task['targets']:
        if mark['camera']==CAMERAS[0]:overlay.rectangle(mark['box'],outline='red',width=3)
    frames=0;steps=0;latencies=[];states=[sim.state()];targets=[];chunks=[];clipped=0
    stop_frame=min(len(gt)-1,args.start_frame+args.max_frames) if args.max_frames else len(gt)-1
    def video_frame(status):
        nonlocal frames
        canvas=Image.new('RGB',(1280,960),'#111824');draw=ImageDraw.Draw(canvas)
        draw.text((16,10),f'take_tube 30k | {args.inference_mode}, denoise {args.denoise_steps}, blend {args.chunk_smoothing_steps} | frame {source.frame} | {status}',font=titlefont,fill='white')
        draw.text((16,43),f'SIM state feedback | recorded RGB + initial box | control steps {steps} | video {frames/30:.2f}s',font=font,fill='#b8d2ee')
        renderer.update_scene(sim.data,camera=camera,scene_option=view)
        canvas.paste(Image.fromarray(renderer.render()),(0,80))
        panels=[('Initial reference + box (frame '+str(source.task['reference_frame'])+')',reference,(784,80),(480,250)),
                ('Current recorded ego (no new box)',Image.fromarray(source.current_images[CAMERAS[0]]),(784,365),(480,250)),
                ('Left wrist',Image.fromarray(source.current_images[CAMERAS[1]]),(784,650),(232,174)),
                ('Right wrist',Image.fromarray(source.current_images[CAMERAS[2]]),(1024,650),(232,174))]
        for title,rgb,(x,y),size in panels:
            rgb=rgb.copy();rgb.thumbnail(size);canvas.paste(rgb,(x,y+26));draw.text((x,y),title,font=font,fill='#ffdd66')
        error=np.max(np.abs(sim.state()[ARM_INDICES]-sim.held))
        draw.text((16,675),f'Arm/gripper target error (max): {error:.3f} rad',font=font,fill='#8ce8b6')
        draw.text((16,710),'Controller: XR arm PD + deployed Dex1 gravity FF; lower body fixed.',font=font,fill='#b8d2ee')
        draw.text((16,743),'Model input state: simulated achieved body29 + Dex1 positions.',font=font,fill='#b8d2ee')
        draw.text((16,776),'Only the initial state is taken from the dataset; later recorded states are NOT inputs.',font=font,fill='#b8d2ee')
        draw.text((16,865),'Recorded RGB is independent of robot motion. No calibrated tube/rack contact or liquid physics.',font=font,fill='#e0bb83')
        draw.text((16,900),('IDEAL CLOCK: physics/video/history freeze while inference is pending; compute time excluded.'
                  if args.inference_clock=='ideal' else
                  'Timing includes simulated hold during model compute; excludes network and rendering overhead.'),font=font,fill='#e0bb83')
        encoder.stdin.write(np.asarray(canvas).tobytes())
        if frames==0:canvas.save(out/'first_frame.png')
        frames+=1
    try:
        worker.request(dict(op='reset',episode=episode,start_frame=args.start_frame,state=sim.state().tolist()))
        observations=[];video_frame('initial')
        while args.inference_mode=='sync' and source.frame<stop_frame:
            response=worker.request(dict(op='predict',observations=observations,denoise_steps=args.denoise_steps))
            if response.get('ok') is False:raise RuntimeError(response)
            assert response['frame']==source.frame
            actions=np.asarray(response['actions']);assert actions.shape==(32,16) and np.isfinite(actions).all()
            latency=response['inference_seconds'];latencies.append(latency)
            chunks.append(dict(frame=source.frame,actions=actions.tolist(),inference_seconds=latency))
            # Match the real controller's continuous hold while a model request runs.
            for _ in range(int(np.ceil(latency*30))):
                clipped+=sim.step(sim.held);video_frame('inference / HOLD')
            observations=[]
            for target in actions[:min(args.execute,stop_frame-source.frame)]:
                clipped+=sim.step(target);source.advance();steps+=1
                state=sim.state();states.append(state);targets.append(sim.held.copy())
                observations.append(dict(frame=source.frame,state=state.tolist()))
                video_frame('execute prediction')
            print(json.dumps(dict(frame=source.frame,total=stop_frame,steps=steps,inference_seconds=latency)),flush=True)
        if args.inference_mode=='async':
            source.task['frames']=stop_frame+1
            class SimPump:
                def __init__(inner):inner.closed=False;inner.deadline=time.monotonic()
                def snapshot(inner):return sim.state()
                def target_snapshot(inner):return sim.held.copy()
                def enable(inner):pass
                def close(inner):inner.closed=True
                def execute(inner,actions,*,traces):
                    nonlocal clipped
                    assert not inner.closed
                    result=[]
                    for target in actions:
                        clipped+=sim.step(target)
                        inner.deadline=max(inner.deadline+1/30,time.monotonic())
                        time.sleep(max(0,inner.deadline-time.monotonic()))
                        result.append(sim.state())
                    return result
            def call(packet):
                op=packet['op']
                if op=='reset':
                    return worker.request(dict(op='reset',episode=episode,start_frame=args.start_frame,state=packet['state']))
                if op=='observe_batch':
                    return worker.request(dict(op='observe_only',observations=[
                        dict(frame=p['dataset_frame'],state=p['state']) for p in packet['observations']]))
                if op=='stop':return worker.request(dict(op='stop'))
                result=worker.request(dict(op='predict',observations=[],denoise_steps=args.denoise_steps))
                latencies.append(result['inference_seconds'])
                chunks.append(dict(frame=result['frame'],actions=result['actions'],inference_seconds=result['inference_seconds'],
                                   pipeline_seconds=result['pipeline_seconds']))
                result['binding_id']='simulation-client'
                return result
            def tick(packet):
                nonlocal steps
                if packet['index']>steps:
                    states.append(packet['state']);targets.append(packet['action'])
                steps=packet['index']
                video_frame('async HOLD' if packet['holding'] else 'async execute / prefetch')
            async_summary=run_session_async(source,SimPump(),call,binding='simulation-client',
                horizon=args.inference_horizon,delay_steps=args.inference_delay_steps,
                expected_denoise_steps=args.denoise_steps,smoothing_steps=args.chunk_smoothing_steps,on_tick=tick,encode_images=lambda images:None,
                simulate_latency=True,freeze_simulation_on_inference=args.inference_clock=='ideal',
                report=lambda line:print(line,flush=True))
    finally:
        encoder.stdin.close();encoder.wait(timeout=60);renderer.close();source.close()
    if encoder.returncode:raise RuntimeError('Video encoding failed')
    np.savez_compressed(out/'trajectory.npz',sim_state=np.asarray(states),targets=np.asarray(targets))
    (out/'predicted_chunks.json').write_text(json.dumps(chunks))
    summary=dict(model=args.variant,checkpoint=worker.checkpoint,episode=episode,start_frame=args.start_frame,
        final_frame=source.frame,executed_steps=(async_summary["execution_steps"] if args.inference_mode=="async" else steps),video_frames=frames,fps=30,video_seconds=frames/30,
        mean_inference_seconds=float(np.mean(latencies)),replan_steps=(args.execute if args.inference_mode=="sync" else None),clipped_targets=clipped,
        timeline_steps=steps,model_action_steps=(async_summary["execution_steps"] if args.inference_mode=="async" else steps),
        inference_horizon=args.inference_horizon,inference_delay_steps=args.inference_delay_steps,
        inference_mode=args.inference_mode,denoise_steps=args.denoise_steps,chunk_smoothing_steps=args.chunk_smoothing_steps,
        inference_clock=args.inference_clock,
        async_stats=async_summary if args.inference_mode=='async' else None,
        state_input='achieved MuJoCo body29 + gripper positions; recorded state used only for initial reset',
        vision_input='current recorded RGB and exact annotated reference image/initial box via deployed RPC',
        physics='fixed base, XR PD, shared deployed Dex1 gravity model; no contact scene/task-success metric',
        gravity=sim.gravity.summary(),robot_io=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)


def rollout(args, row, worker, out):
    out.mkdir(parents=True, exist_ok=False)
    root = Path(row['dataset'])
    info = json.loads((root/'meta/info.json').read_text())
    names = info['features']['observation.state']['names']
    ep = row['episode_index']
    fields = dict(episode_index=ep,episode_chunk=ep//info['chunks_size'])
    gt = np.asarray(pq.read_table(root/info['data_path'].format(**fields),
                               columns=['observation.state'])['observation.state'].to_pylist(),np.float32)
    active = action_mask(row['skill'],row['hand'])
    sim = Physics(out,Path(args.asset),names,gt[row['start_frame']],active)
    renderer = mujoco.Renderer(sim.model,height=576,width=768)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [.15,0,1.13]
    camera.distance,camera.azimuth,camera.elevation = 1.55,135,-13
    view = mujoco.MjvOption()
    view.geomgroup[3:]=0
    captures = [cv2.VideoCapture(str(root/info['video_path'].format(video_key='observation.images.'+c,**fields)))
                for c in ('ego_view','left_wrist','right_wrist')]
    encoder = subprocess.Popen([args.ffmpeg,'-hide_banner','-loglevel','error','-y',
        '-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1280x768','-r','30','-i','-',
        '-an','-c:v','libx264','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'simulation.mp4')],stdin=subprocess.PIPE)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
    titlefont = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',22)
    total = min(len(gt), args.max_frames or len(gt))
    trajectory,targets,latencies,requests = [sim.state()],[],[],[]
    clipped = 0
    def frame(t):
        im = Image.new('RGB',(1280,768),'#111824')
        renderer.update_scene(sim.data,camera=camera,scene_option=view)
        im.paste(Image.fromarray(renderer.render()),(0,80))
        draw = ImageDraw.Draw(im)
        draw.text((16,10),f'{args.variant} | {row["skill"]} | episode {ep}',font=titlefont,fill='white')
        draw.text((16,42),f'Physics state feedback + recorded RGB | {t/30:.2f}s / {(total-1)/30:.2f}s | hand: {row["hand"]}',font=font,fill='#b8d2ee')
        for k,cap in enumerate(captures):
            cap.set(cv2.CAP_PROP_POS_FRAMES,t)
            ok,arr = cap.read()
            if not ok:
                raise RuntimeError('Recorded preview frame missing')
            rgb = Image.fromarray(cv2.cvtColor(arr,cv2.COLOR_BGR2RGB))
            if k == 0 and t == row['start_frame']:
                overlay=ImageDraw.Draw(rgb)
                for mark in row['targets']:
                    if mark.get('camera')=='observation.images.ego_view':
                        overlay.rectangle(mark['box'],outline='red',width=3)
            rgb.thumbnail((480,192))
            x,y=784+(480-rgb.width)//2,80+k*202
            im.paste(rgb,(x,y))
            draw.text((790,y+2),('Recorded ego','Recorded left wrist','Recorded right wrist')[k],font=font,fill='#ffdd66')
        s=sim.state()
        draw.text((16,667),f'Dex1 motor (sim / recorded): left {s[29]:.2f} / {gt[t,29]:.2f}   right {s[30]:.2f} / {gt[t,30]:.2f} rad',font=font,fill='#8ce8b6')
        draw.text((16,700),'Fixed pelvis; diagnostic PD + arm gravity compensation; no calibrated objects or liquid simulation.',font=font,fill='#e0bb83')
        draw.text((16,730),'Recorded RGB does not respond to simulated actions. This video is NOT a grasp-success evaluation.',font=font,fill='#e0bb83')
        encoder.stdin.write(np.asarray(im).tobytes())
        if t in (0,total//2,total-1):
            im.save(out/f'frame_{t:04d}.png')
    try:
        worker.request(dict(op='reset',episode=ep,state=sim.state().tolist()))
        t=row['start_frame']
        frame(t)
        observations=[]
        while t < total-1:
            response=worker.request(dict(op='predict',observations=observations,denoise_steps=args.denoise_steps))
            if response['frame'] != t:
                raise RuntimeError('Policy/simulator frame mismatch')
            chunk=np.asarray(response['actions'])
            latencies.append(response['inference_seconds'])
            requests.append(dict(frame=t,chunk=chunk.tolist()))
            observations=[]
            for target in chunk[:min(args.execute,total-1-t)]:
                clipped+=sim.step(target)
                t+=1
                state=sim.state()
                trajectory.append(state)
                targets.append(target)
                observations.append(dict(frame=t,state=state.tolist()))
                frame(t)
            print(json.dumps(dict(variant=args.variant,episode=ep,frame=t,total=total,
                                  inference_seconds=latencies[-1])),flush=True)
    finally:
        encoder.stdin.close()
        encoder.wait(timeout=60)
        renderer.close()
        for cap in captures:
            cap.release()
    if encoder.returncode:
        raise RuntimeError('Video encoding failed')
    trajectory=np.asarray(trajectory)
    np.savez_compressed(out/'trajectory.npz',sim_state=trajectory,targets=np.asarray(targets),gt_state=gt[:len(trajectory)])
    (out/'predicted_chunks.json').write_text(json.dumps(requests))
    arm_dims=[i for i in range(16) if i not in (7,15) and active[i]]
    hand_dims=[i for i in (7,15) if active[i]]
    error=np.abs(trajectory[:,ARM_INDICES]-gt[:len(trajectory),ARM_INDICES])
    summary=dict(model=args.variant,checkpoint=CHECKPOINTS[args.variant],episode=ep,skill=row['skill'],hand=row['hand'],
        split='val',frames=len(trajectory),fps=30,replan_frames=args.execute,denoise_steps=20,
        active_arm_mae_rad=float(error[:,arm_dims].mean()),active_gripper_mae_rad=float(error[:,hand_dims].mean()),
        clipped_targets=clipped,mean_inference_seconds=float(np.mean(latencies)),
        state_source='achieved MuJoCo state after initial recorded-state reset; no future recorded state input',
        vision='recorded three-camera RGB; initial marks only; no policy-generated or simulator RGB input',
        scene='official G1/Dex1; fixed base; no calibrated object/contact scene; no success metric',
        dex1_mapping='URDF outward sliders [-0.020,0.0245] m mapped to motor [0,5.4] rad; approximate',
        asset_revision='deeb607f3a8945a394f53068c28b3a88dde7ff06',mujoco_version=mujoco.__version__)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--variant',choices=CHECKPOINTS,required=True)
    ap.add_argument('--gpu',type=int,default=0)
    ap.add_argument('--episodes',type=int,nargs='+',required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--asset',default=DEFAULT_ASSET)
    ap.add_argument('--ffmpeg',required=True)
    ap.add_argument('--execute',type=int,default=8)
    ap.add_argument('--max-frames',type=int,default=0)
    ap.add_argument('--rpc-host',help='Reuse an existing deployed model on this SSH host; no robot I/O')
    ap.add_argument('--rpc-root',default='/home/yanjun/ykj/labwam_g1')
    ap.add_argument('--dataset',type=Path)
    ap.add_argument('--start-frame',type=int,default=60)
    ap.add_argument('--inference-mode',choices=['sync','async'],default='sync')
    ap.add_argument('--denoise-steps',type=int,default=20)
    ap.add_argument('--chunk-smoothing-steps',type=int,default=0)
    ap.add_argument('--inference-clock',choices=['measured','ideal'],default='measured',
                    help='ideal freezes physics/video/history during unavailable inference; offline RPC async only')
    ap.add_argument('--inference-horizon',type=int,default=32)
    ap.add_argument('--inference-delay-steps',type=int,default=16)
    args=ap.parse_args()
    if args.inference_clock=='ideal' and (not args.rpc_host or args.inference_mode!='async'):
        ap.error('ideal inference clock requires the offline --rpc-host async preview')
    if not 1<=args.execute<=32:
        ap.error('execute must be 1..32')
    args.output.mkdir(parents=True,exist_ok=False)
    if args.rpc_host:
        if args.dataset is None or args.variant != 'take_tube':
            ap.error('Deployed RPC preview requires --dataset and --variant take_tube')
        worker=DeploymentWorker(args,args.output/'model_worker.log')
        try:
            for ep in args.episodes:deployment_rollout(args,worker,args.output/f'episode_{ep:06d}',ep)
        finally:worker.close()
        return
    rows=json.loads((ROOT/f'data/boxed_v3_suite/{args.variant}.json').read_text())['segments']
    worker=Worker(args.variant,args.gpu,args.output/'model_worker.log')
    try:
        for ep in args.episodes:
            row=next(r for r in rows if r['episode_index']==ep and r['split']=='val')
            rollout(args,row,worker,args.output/f'episode_{ep:06d}')
    finally:
        worker.close()


if __name__=='__main__':
    main()
