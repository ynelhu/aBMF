#!/usr/bin/env python3
"""CU MATLAB trial arrays, mixed-gamma checks, statistics, retries and exports."""
import argparse, collections, csv, datetime, fcntl, hashlib, json, math, os
from pathlib import Path
import shutil, signal, statistics, subprocess, sys, uuid, zipfile

ROOT=Path(__file__).resolve().parents[1]
def now():return datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.'+uuid.uuid4().hex+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8');os.replace(str(tmp),str(p))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def csvread(p):
    with Path(p).open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def csvwrite(p,data,fields):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.'+uuid.uuid4().hex+'.tmp')
    with tmp.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(data)
    os.replace(str(tmp),str(p))
def close(a,b):return math.isfinite(a) and math.isfinite(b) and abs(a-b)<=1e-9*max(1,abs(b))

def audit(output,unit,manifest):
    config=manifest['package'];suite=manifest['suite'];kind=config['kind']
    raw=csvread(output/'raw_trial_results.csv')
    if len(raw)!=unit['expected_records']:raise ValueError('Expected %d records, got %d.'%(unit['expected_records'],len(raw)))
    keys=['graph_name','rank','method','trial'] if kind=='maxcut' else ['sample_index','epsilon','rank','trial']+(['gamma_multiplier'] if suite=='sensitivity' else ['method'])
    seen=set()
    for row in raw:
        key=tuple(row[k] for k in keys)
        if key in seen:raise ValueError('Duplicate experimental record: '+str(key))
        seen.add(key)
        direct=kind=='maxcut' and row['method']=='Full-SDP-PG'
        if not direct and int(float(row['trial']))!=unit['trial']:raise ValueError('Trial ID changed.')
        required=['time','iterations']+(['best_cut','sdp_objective','equality_residual'] if kind=='maxcut' else ['surrogate_objective','complementarity_residual','rel_disagreement'])
        if any(not math.isfinite(float(row[k])) for k in required):raise ValueError('Nonfinite experimental metric.')
        if not direct and kind=='maxcut':
            indices={('G55',50):1,('G67',50):2,('G67',100):3,('tiny_gset',2):1}
            index=indices[(row['graph_name'],int(float(row['rank'])))]
            if float(row['initialization_seed'])!=100000*index+unit['trial']:raise ValueError('MaxCut initialization seed changed.')
            if float(row['rounding_seed'])!=200000*index+unit['trial']:raise ValueError('MaxCut rounding seed changed.')
        elif kind=='nn':
            samples=config['selected_test_indices'] if unit['profile']=='standard' else [config['selected_test_indices'][1]]
            eps=[.6,.8] if unit['profile']=='standard' else [.8];ranks=[8,16] if unit['profile']=='standard' else [8]
            sp=samples.index(int(float(row['sample_index'])))+1
            ei=next(i+1 for i,x in enumerate(eps) if abs(float(row['epsilon'])-x)<1e-10)
            ri=ranks.index(int(float(row['rank'])))+1
            expected=314159+100000*sp+10000*ei+100*ri+unit['trial']
            if float(row['initialization_seed'])!=expected:raise ValueError('NN paired seed changed.')
    histories=sorted((output/'histories').glob('*aBMF-our*.csv')) if kind=='nn' else sorted(output.glob('*aBMF-our*.csv'))
    expected=0 if suite=='sensitivity' else unit['conditions']
    if len(histories)!=expected:raise ValueError('Expected %d our histories, got %d.'%(expected,len(histories)))
    checked=0;increases=0;decreases=0
    for p in histories:
        h=csvread(p);previous_next=None
        for k,row in enumerate(h,1):
            if int(float(row['iteration']))!=k:raise ValueError('Incorrect history numbering.')
            used=float(row['gamma_used']);updated=float(row['gamma_next'])
            if not close(used,float(row['gamma_previous'])) or used<=0 or updated<=0:raise ValueError('Invalid gamma.')
            if previous_next is not None and not close(used,previous_next):raise ValueError('Gamma recurrence failed.')
            if float(row['D_zero'])!=0:
                if not close(updated,used):raise ValueError('D=0 must retain previous gamma.')
            else:
                check=max(float(row['directional_quotient'])/(2*manifest['nu']),0)+manifest['epsilon0']
                hat=max(float(row['loss_gradient_fro'])/2,0)
                expected_value=max(min(used,hat),check)
                if not all([close(check,float(row['gamma_check'])),close(hat,float(row['gamma_hat'])),close(updated,expected_value)]):raise ValueError('Mixed gamma formula mismatch.')
            increases+=updated>used+1e-9*max(1,abs(used));decreases+=updated<used-1e-9*max(1,abs(used))
            previous_next=updated;checked+=1
    return {'passed':True,'our_histories':len(histories),'iterations_checked':checked,'increases':increases,'decreases':decreases,'fixed_gamma_sensitivity':suite=='sensitivity'}

def start(args):
    package=read(ROOT/'PACKAGE.json')
    if package['kind']=='maxcut' and args.suite!='main':raise ValueError('MaxCut has only the main suite.')
    run=(args.run_dir or ROOT/'results'/(args.suite+'_'+args.profile+'_'+now())).expanduser().resolve()
    run.mkdir(parents=True,exist_ok=False);(run/'slurm').mkdir()
    snapshot=run/'code_snapshot';snapshot.mkdir()
    candidates=[p for p in ROOT.iterdir() if p.is_file() and p.suffix in ('.sh','.m','.json','.txt')]
    for name in ('src','hpc'):candidates.extend(p for p in (ROOT/name).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    for p in candidates:
        dest=snapshot/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    standard=args.profile=='standard';trials=10 if standard else 1
    conditions=(3 if package['kind']=='maxcut' else 12) if standard else 1
    records=12 if package['kind']=='maxcut' else 48
    if args.suite=='sensitivity':records=60
    if not standard:records=5 if package['kind']=='maxcut' else 3 if args.suite=='sensitivity' else 4
    units=[{'index':i,'trial':i+1,'profile':args.profile,'suite':args.suite,'conditions':conditions,'expected_records':records} for i in range(trials)]
    manifest={'schema':1,'created_utc':now(),'package':package,'profile':args.profile,'suite':args.suite,'units':units,
        'nu':args.nu,'epsilon0':args.epsilon0,'python':sys.executable,'jobs':[],'pack_jobs':[],'unit_jobs':{},
        'resources':{'partition':args.partition,'cpus':args.cpus,'mem':args.mem,'time':args.time,'max_parallel':args.max_parallel,'matlab_module':'matlab-R2022b'},
        'source_sha256':{str(p.relative_to(snapshot)):sha(p) for p in sorted(snapshot.rglob('*')) if p.is_file()}}
    write(run/'manifest.json',manifest)
    for u in units:write(run/'unit_configs'/('%03d.json'%u['index']),dict(u,nu=args.nu,epsilon0=args.epsilon0,kind=package['kind']))
    if not args.dry_run:submit(run,manifest,list(range(trials)))
    else:print('DRY RUN: no Slurm submission or MATLAB execution.')
    print('Experiment:',package['name'],'Suite:',args.suite,'Profile:',args.profile)
    print('Units:',trials,'Expected method records:',records*trials,'Python:',sys.executable)
    print('Run:',run);print('Result packs:',run/'exports')

def submit(run,m,indices):
    if not indices:print('All trials already completed.');return
    if not shutil.which('sbatch'):raise RuntimeError('sbatch unavailable; launch on the CU supercomputer or use --dry-run.')
    r=m['resources'];code=run/'code_snapshot';name=m['package']['name']
    common=['--parsable','--partition='+r['partition']]
    cmd=['sbatch']+common+['--job-name='+name,'--array='+','.join(map(str,indices))+'%'+str(r['max_parallel']),
        '--ntasks=1','--cpus-per-task='+str(r['cpus']),'--mem='+r['mem'],'--time='+r['time'],'--chdir='+str(code),
        '--output='+str(run/'slurm/array_%A_%a.log'),'--error='+str(run/'slurm/array_%A_%a.err'),str(code/'hpc/array.sh'),sys.executable,str(run)]
    job=subprocess.check_output(cmd,text=True).strip().split(';')[0]
    if not job.isdigit():raise RuntimeError('Unexpected sbatch response: '+job)
    m['jobs'].append(job)
    for i in indices:m['unit_jobs'][str(i)]=job+'_'+str(i)
    write(run/'manifest.json',m)
    cmd=['sbatch']+common+['--job-name='+name+'_pack','--ntasks=1','--cpus-per-task=1','--mem=4G','--time=00:30:00',
        '--dependency=afterany:'+job,'--output='+str(run/'slurm/pack_%j.log'),'--error='+str(run/'slurm/pack_%j.err'),str(code/'hpc/collect.sh'),sys.executable,str(run)]
    try:
        m['pack_job']=subprocess.check_output(cmd,text=True).strip().split(';')[0];m.setdefault('pack_jobs',[]).append(m['pack_job']);print('Array:',job,'Automatic statistics/pack:',m['pack_job'])
    except subprocess.CalledProcessError:
        m['pack_submit_failed']=True;print('Trials submitted; pack submission failed. Use PACK_RESULTS.sh manually.',file=sys.stderr)
    write(run/'manifest.json',m)

def worker(run,index):
    m=read(run/'manifest.json');unit=m['units'][index];base=run/'units'/('unit_%03d'%index)
    attempt=base/('attempt_'+now()+'_'+uuid.uuid4().hex[:8]);attempt.mkdir(parents=True)
    state={'status':'running','started_utc':now(),'index':index,'attempt':str(attempt.relative_to(run)),'host':os.uname().nodename}
    write(base/'state.json',state);code=run/'code_snapshot';proc=None
    def interrupted(signum,frame):
        if proc is not None and proc.poll() is None:os.killpg(proc.pid,signal.SIGTERM)
        raise RuntimeError('Worker interrupted by signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        for n,value in m['source_sha256'].items():
            if sha(code/n)!=value:raise ValueError('Code/data snapshot changed: '+n)
        output=attempt/'output'
        env=dict(os.environ,ABMF_CODE_DIR=str(code),ABMF_UNIT_JSON=str(run/'unit_configs'/('%03d.json'%index)),ABMF_OUTPUT_DIR=str(output))
        with (attempt/'engine.log').open('w') as log:
            proc=subprocess.Popen(['bash',str(code/'hpc/matlab_worker.sh')],cwd=str(attempt),env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            rc=proc.wait()
        if rc:raise RuntimeError('MATLAB exit code %d; inspect %s'%(rc,attempt/'engine.log'))
        check=audit(output,unit,m);write(attempt/'GAMMA_CHECK.json',check)
        state.update(status='completed',finished_utc=now());write(base/'state.json',state);print('Trial',unit['trial'],'completed; gamma checks passed.');return 0
    except Exception as exc:
        state.update(status='failed',finished_utc=now(),error=str(exc));write(base/'state.json',state);print(str(exc),file=sys.stderr);return 1

def sched_states(m):
    if not m['jobs'] or not shutil.which('sacct'):return {}
    try:
        out=subprocess.check_output(['sacct','-X','-n','-P','-j',','.join(m['jobs']),'--format=JobID%64,State%32'],text=True,stderr=subprocess.DEVNULL,timeout=20)
        return {x.split('|')[0].strip():x.split('|')[1].split()[0].rstrip('+') for x in out.splitlines() if '|' in x}
    except (subprocess.SubprocessError,OSError):return {}

def aggregate(raw,keys,metrics):
    groups=collections.defaultdict(list)
    for row in raw:groups[tuple(row[k] for k in keys)].append(row)
    result=[]
    for key,group in sorted(groups.items()):
        row=dict(zip(keys,key));row['num_trials']=len(group)
        for metric in metrics:
            vals=[]
            for r in group:
                try:value=float(r.get(metric,'nan'))
                except ValueError:value=float('nan')
                if math.isfinite(value):vals.append(value)
            row[metric+'_mean']=statistics.mean(vals) if vals else float('nan')
            row[metric+'_std']=statistics.stdev(vals) if len(vals)>1 else 0 if vals else float('nan')
        result.append(row)
    return result

def collect(run):
    m=read(run/'manifest.json');kind=m['package']['kind'];suite=m['suite'];sched=sched_states(m);raw=[];statuses=[];checks=[]
    bad={'FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE'}
    for unit in m['units']:
        p=run/'units'/('unit_%03d'%unit['index'])/'state.json';state=read(p) if p.exists() else {'status':'not_started'}
        status=state['status'];scheduler=sched.get(m['unit_jobs'].get(str(unit['index']),''),'')
        if status!='completed' and scheduler in bad:status=scheduler.lower()
        elif status=='not_started' and scheduler in ('PENDING','RUNNING','CONFIGURING'):status=scheduler.lower()
        elif status in ('running','not_started') and scheduler=='COMPLETED':status='missing_output'
        if status=='completed':
            try:
                output=run/state['attempt']/'output';check=audit(output,unit,m);checks.append(dict(unit_index=unit['index'],**check))
                for row in csvread(output/'raw_trial_results.csv'):row['unit_index']=unit['index'];raw.append(row)
            except Exception as exc:status='invalid_output';state['error']=str(exc)
        statuses.append(dict(unit,status=status,scheduler_state=scheduler,attempt=state.get('attempt',''),error=state.get('error','')))
    folder=run/'statistics';folder.mkdir(exist_ok=True)
    csvwrite(folder/'status.csv',statuses,list(m['units'][0])+['status','scheduler_state','attempt','error'])
    if kind=='maxcut':
        keys=['graph_name','n','m','rank','method'];metrics=['iterations','time','final_gamma','final_gamma_next','final_rho','sdp_objective','bilinear_sdp_objective','best_cut','mean_cut','cut_ratio','rel_disagreement','equality_residual','multiplier_norm_inf','gamma_candidate','gradient_fro','gamma_cap_hit','rho_cap_hit']
    else:
        keys=['sample_index','epsilon','rank']+(['gamma_multiplier'] if suite=='sensitivity' else ['method'])
        metrics=['iterations','time','final_gamma','final_gamma_next','final_gamma_hat','surrogate_objective','margin_surrogate','affine_residual','inequality_residual','complementarity_residual','rel_disagreement','psd_violation','success']
        if suite=='sensitivity':metrics=['gamma_reference','gamma_value']+metrics
    raw_fields=list(raw[0]) if raw else ['unit_index']+keys+['trial']+metrics
    csvwrite(folder/'raw_results.csv',raw,raw_fields)
    summary=aggregate(raw,keys,metrics);fields=keys+['num_trials']+[x+s for x in metrics for s in ('_mean','_std')]
    csvwrite(folder/'summary_mean_std.csv',summary,fields)
    expected=len(m['units']);coverage=[]
    for row in summary:
        entry={k:row[k] for k in keys};entry.update(n_expected=expected,n_completed=row['num_trials'],n_missing=expected-row['num_trials']);coverage.append(entry)
    csvwrite(folder/'coverage.csv',coverage,keys+['n_expected','n_completed','n_missing'])
    if kind=='nn' and suite=='main':
        overall=aggregate(raw,['method'],metrics)
        for row in overall:row['num_runs']=row.pop('num_trials');row['success_rate']=row['success_mean']
        csvwrite(folder/'rebuttal_aggregate.csv',overall,['method','num_runs']+[x+s for x in metrics for s in ('_mean','_std')]+['success_rate'])
    elif kind=='nn' and suite=='sensitivity':
        overall=aggregate(raw,['gamma_multiplier'],metrics)
        csvwrite(folder/'gamma_sensitivity_aggregate.csv',overall,['gamma_multiplier','num_trials']+[x+s for x in metrics for s in ('_mean','_std')])
    counts=dict(collections.Counter(r['status'] for r in statuses));complete=counts.get('completed',0)==len(m['units'])
    stops=dict(collections.Counter(r.get('stop_reason','') for r in raw if r.get('method')=='aBMF-our'))
    handoff={'created_utc':now(),'experiment':m['package']['name'],'suite':suite,'profile':m['profile'],'status_counts':counts,
        'all_units_completed':complete,'partial':not complete,'method_records':len(raw),'expected_method_records':sum(u['expected_records'] for u in m['units']),
        'gamma_rule':'max(min(previous,hat),check); exact D=0 retains previous','nu':m['nu'],'epsilon0':m['epsilon0'],'sigma_h':0,
        'gamma_checks':checks,'our_stop_reasons':stops,'note':'Job completion/formula checks do not establish feasibility, stationarity, or performance improvement.'}
    write(folder/'HANDOFF.json',handoff);print('Run:',run);print('Status:',counts,'Records: %d/%d'%(len(raw),handoff['expected_method_records']));return handoff

def pack(run):
    handoff=collect(run);folder=run/'exports';folder.mkdir(exist_ok=True)
    p=folder/(handoff['experiment']+'_'+('complete_' if handoff['all_units_completed'] else 'partial_')+now()+'_'+uuid.uuid4().hex[:6]+'.zip')
    inventory={}
    with zipfile.ZipFile(p,'w',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(run.rglob('*')):
            rel=f.relative_to(run)
            if not f.is_file() or rel.parts[0]=='exports' or '__pycache__' in rel.parts or f.name=='.statistics.lock' or f.suffix=='.tmp':continue
            data=f.read_bytes();name=str(Path('ABMF_RESULT')/rel);z.writestr(name,data);inventory[name]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
        z.writestr('ABMF_RESULT/FILES.json',json.dumps(inventory,indent=2))
    p.with_suffix('.zip.sha256').write_text(sha(p)+'  '+p.name+'\n');print('Upload result pack:',p)

def main():
    p=argparse.ArgumentParser(description=__doc__);sp=p.add_subparsers(dest='command')
    a=sp.add_parser('start');a.add_argument('--profile',choices=['smoke','standard'],default='standard');a.add_argument('--suite',choices=['main','sensitivity'],default='main')
    a.add_argument('--run-dir',type=Path);a.add_argument('--partition',default='CU');a.add_argument('--cpus',type=int,default=64);a.add_argument('--mem',default='64G')
    a.add_argument('--time',default='4800:00:00');a.add_argument('--max-parallel',type=int,default=1);a.add_argument('--nu',type=float,default=.9);a.add_argument('--epsilon0',type=float,default=1e-6);a.add_argument('--dry-run',action='store_true')
    for name in ('status','pack','retry','cancel'):sp.add_parser(name).add_argument('run_dir',type=Path)
    w=sp.add_parser('worker');w.add_argument('run_dir',type=Path);w.add_argument('index',type=int)
    args=p.parse_args()
    if not args.command:p.error('a command is required')
    if args.command=='start':
        if not (0<args.nu<1 and math.isfinite(args.nu) and args.epsilon0>0 and math.isfinite(args.epsilon0) and args.cpus>0 and args.max_parallel>0):p.error('Invalid nu/epsilon0/resources')
        start(args);return 0
    run=args.run_dir.expanduser().resolve()
    if args.command=='worker':return worker(run,args.index)
    with (run/'.statistics.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        if args.command=='status':collect(run)
        elif args.command=='pack':pack(run)
        elif args.command=='cancel':
            m=read(run/'manifest.json');ids=m['jobs']+m.get('pack_jobs',[])
            if ids:subprocess.run(['scancel']+ids,check=True)
            print('Cancelled only this run:',','.join(ids))
        else:
            m=read(run/'manifest.json')
            if m['jobs'] and shutil.which('squeue'):
                q=subprocess.check_output(['squeue','-h','-u',os.environ.get('USER',''),'-o','%i'],text=True)
                if {x.strip().split('_')[0] for x in q.splitlines()}&set(m['jobs']):raise RuntimeError('Trials are still active; retry after they exit.')
            collect(run);indices=[int(x['index']) for x in csvread(run/'statistics/status.csv') if x['status']!='completed'];submit(run,m,indices)
    return 0
if __name__=='__main__':
    try:sys.exit(main())
    except Exception as exc:print('ERROR:',str(exc),file=sys.stderr);sys.exit(1)
