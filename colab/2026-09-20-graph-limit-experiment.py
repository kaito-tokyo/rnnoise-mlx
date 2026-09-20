# Compare CUDA graph limits in isolated, reproducible subprocesses.
import os, sys, json, time, subprocess
from pathlib import Path
assert FEATURE_IDENTITY == "3dae648bb24075075cccfaa0b22020b5075859506f841bae52429520c0f1ccc4"
graph_experiment = Path("/content/runs") / time.strftime("graph-limit-bs4-3-%Y%m%d-%H%M%S")
graph_experiment.mkdir(parents=True, exist_ok=False)
wrapper = r'''
import os, sys, json, time, runpy
from pathlib import Path
import mlx.core as mx
from rnnoise_mlx.training.config import TrainConfig
from rnnoise_mlx.training_cuda import CUDATrainingLoop
mx.random.seed(141)
events = Path(sys.argv[1])
sys.argv = ["/content/rnnoise-mlx/colab/profile_train.py"] + sys.argv[2:]
original = CUDATrainingLoop.run_update
counter = 0
def wrapped(self, *args, **kwargs):
    global counter
    counter += 1
    with events.open("a") as f:
        f.write(json.dumps({"event":"start","update":counter,"time":time.time()})+"\n")
    result = original(self,*args,**kwargs)
    with events.open("a") as f:
        f.write(json.dumps({"event":"end","update":counter,"time":time.time(),"peak_memory":mx.get_peak_memory()})+"\n")
    return result
CUDATrainingLoop.run_update = wrapped
runpy.run_path(sys.argv[0],run_name="__main__")
'''
(graph_experiment/"wrapper.py").write_text(wrapper)
driver = r'''
import os,sys,json,time,subprocess,statistics
from pathlib import Path
import psutil
root=Path(sys.argv[1]); identity=sys.argv[2]
results=[]
for limit in (20,100):
    out=root/f"ops-{limit}"; out.mkdir()
    env=os.environ.copy(); env["MLX_MAX_OPS_PER_BUFFER"]=str(limit)
    cmd=[sys.executable,str(root/"wrapper.py"),str(out/"events.jsonl"),
         "--features","/content/train.npy","--output",str(out/"training"),
         "--batch-size","4","--max-updates","3","--feature-identity",identity]
    (out/"config.json").write_text(json.dumps({"command":cmd,"seed":141,"graph_env":{k:env.get(k) for k in ("MLX_MAX_OPS_PER_BUFFER","MLX_MAX_MB_PER_BUFFER","MLX_CUDA_GRAPH_CACHE_SIZE","MLX_USE_CUDA_GRAPHS")}},indent=2))
    samples=[]
    with (out/"training.log").open("w") as log:
        p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
        (out/"pid").write_text(str(p.pid))
        proc=psutil.Process(p.pid); proc.cpu_percent()
        while p.poll() is None:
            try:
                cpu=proc.cpu_percent()
                gpu=subprocess.run(["nvidia-smi","--query-gpu=utilization.gpu,memory.used","--format=csv,noheader,nounits"],capture_output=True,text=True)
                vals=[float(v.strip()) for v in gpu.stdout.strip().split(",")] if gpu.returncode==0 else [None,None]
                samples.append({"time":time.time(),"cpu_percent":cpu,"gpu_percent":vals[0],"gpu_memory_mib":vals[1]})
            except psutil.NoSuchProcess: break
            time.sleep(0.5)
        rc=p.wait()
    (out/"samples.json").write_text(json.dumps(samples))
    result={"limit":limit,"returncode":rc}
    summary=out/"training/training_summary.json"
    if summary.exists(): result["training"]=json.loads(summary.read_text())
    results.append(result)
    (out/"exit.json").write_text(json.dumps({"returncode":rc}))
    print(json.dumps(result),flush=True)
    if rc: break
(root/"results.json").write_text(json.dumps(results,indent=2))
(root/"completed.json").write_text(json.dumps({"ok":len(results)==2 and all(r["returncode"]==0 for r in results)}))
'''
(graph_experiment/"driver.py").write_text(driver)
with (graph_experiment/"driver.log").open("w") as log:
    graph_driver = subprocess.Popen([sys.executable,str(graph_experiment/"driver.py"),str(graph_experiment),FEATURE_IDENTITY],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(graph_experiment/"driver.pid").write_text(str(graph_driver.pid))
print(json.dumps({"run_dir":str(graph_experiment),"pid":graph_driver.pid}))
