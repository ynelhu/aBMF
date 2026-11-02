2026-11-01 正式版 v2 修复说明
Slurm 会将 sbatch 脚本复制到临时目录。旧版由脚本自身路径定位 Python 控制器，
因此可能报错 /var/spool/slurm/d/hpc/cu_hpc.py 不存在，实验和自动打包均未启动。
本版从提交时传入的绝对 run-dir 定位 code_snapshot，支持脚本复制和任意工作目录。
只修复提交入口；MATLAB 算法、gamma 公式、正式实验参数及原始数据均保持不变。
若旧目录出现上述错误，请使用本包创建新运行，不修改旧 code_snapshot 或 manifest：
  bash RUN_FORMAL.sh --run-dir "$PWD/results/formal_fix_01"
该名称如已存在，请换一个新名称。不要对旧快照直接调用 RETRY_FAILED.sh，
因为旧快照保存的入口脚本仍有同一问题。旧运行可保留用于诊断。
已补测脚本复制到模拟 Slurm spool 后的实验入口和自动打包入口，
包括正式试验记录数量、种子、gamma 检查、统计与导出；此处无真实 MATLAB/CU。

Max-Cut 最新混合 gamma 更新：CU 超算正式运行版

部署与启动
将 aBMF_MaxCut_DynamicMix_CU.zip 上传到 /home/math_t1/enlianghu/，然后执行：
  cd /home/math_t1/enlianghu
  unzip aBMF_MaxCut_DynamicMix_CU.zip
  cd aBMF_MaxCut_DynamicMix_CU
  bash RUN_FORMAL.sh

上述命令直接提交完整正式主实验，不需要另跑 smoke 或手工执行自检。
必要的公式/输出检查会在正式计算任务内部自动完成。
程序自动生成新的结果目录，并在终端打印完整路径和 Slurm 作业号。
如需指定正式结果目录：
  bash RUN_FORMAL.sh --run-dir "$PWD/results/standard_mix_01"
RUN_FORMAL.sh 固定 standard/main，避免误把小规模试跑或敏感性结果当作正式主实验。
资源和 nu、epsilon0 仍可通过同名命令行参数调整。

无需覆盖旧 aBMF 目录，不混入旧结果。每次重新实验请使用新的 run-dir。
默认 CU 分区、1 个任务、每个 array 元素 64 核/64G，申请时间 4800:00:00，
最大并行元素数为 1。按集群上限和需要可修改，例如：
  bash START_ALL.sh --profile standard --cpus 8 --mem 32G --time 72:00:00 \
    --max-parallel 2 --run-dir "$PWD/results/standard_mix_02"
时间参数是申请上限，不是实验完成时间预测；不申请 GPU 或 FAT。

自动使用 ~/miniconda3/bin/python3，不依赖 conda activate；兼容 ~/minconda3 拼写。
可用 PYTHON_BIN 指定解释器。核心提交/统计仅需 Python >=3.8 的标准库。
计算节点按照你的原形式调用：
  module load matlab-R2022b
  matlab -nodisplay -nosplash -nodesktop -r "try, ...; catch ME, ...; exit(1); end; exit(0)"

原实验规模
以本次上传 codes.zip 中的 Max-Cut 程序、数据和已记录配置为基础：
  G55，n=5000，rank=50
  G67，n=10000，rank=50
  G67，n=10000，rank=100
每组重复 10 次，max_iter=3000，tol=1e-6，constraint_tol=1e-6，
共享随机超平面轮数为 100。保留带符号 G67 权重及原 Laplacian 归一化。
比较 sBMF-PGD、sBMF-Mixing、aBMF-our、aBMF-adp，共 120 条方法结果。
Full-SDP 仍按原程序仅在 n<=1200 时运行，正式三组均不会运行此大矩阵基线。
smoke 使用随包 tiny_gset、rank=2、1 次重复、最多 100 轮，包含小图 Full-SDP，
因此 smoke 应有 5 条方法结果。数据已包含，无需下载。

独立调度但保留原种子
正式包按原试验编号 1..10 拆为 10 个 array 元素。
每个元素完成三个条件的同一次试验，不把 trial 重新编号为 1。
初始化仍为 100000*condition_index+trial；舍入仍为 200000*condition_index+trial。
原有条件顺序、初始化、舍入、基线数值更新、评价、停止条件保持原样。

our 的新规则
  check = max(tr(D'*G*D)/(2*nu*||D||_F^2)-sigma_h/4,0)+epsilon0
  hat   = max(0.5*||G||_F-sigma_h/4,0)
  next  = max(min(previous,hat),check)
默认 nu=0.9、epsilon0=1e-6；启动时可以传 --nu 和 --epsilon0。
初始 gamma 沿用原程序 gamma0='auto'；D 恰好为零时保留上一值。
不取方向项绝对值，不强制单调下降，不用 gamma_max 截断 our 的方向保护值。

本问题 h=0，因此 sigma_h=0。使用当前增广拉格朗日子问题的有效梯度：
  G = -L_work/4 + Diag(lambda + rho*(diag(X*Y')-1))。
每轮先执行原有两个块更新，再原样更新 lambda、rho；随后固定新的 lambda、rho，
在当前 X、Y 上计算供下一轮使用的混合 gamma。这一点与下一轮 ALM 目标一致。
lambda、rho 的更新公式、周期及 rho 上限未改。
它是对当前 ALM 子问题的实践更新，不构成动态乘子序列的精确性/收敛证明。

稀疏计算
不形成稠密 n*n 的 G，也不形成 X*Y'。
G 的 Frobenius 范数由稀疏 L 的范数、对角项及交叉项精确计算。
方向商使用归一化 D；复用 L*X、L*Y，AY 型缓存传到下一轮。
很小但非零的 D 使用稀疏 L*U 直接计算，避免差分消去；不会把非零 D 冻结。
history.gamma 和 gamma_used 为当轮实际值；gamma_next 是本轮因子上的新值。
our 的 final_gamma 是最后实际值，final_gamma_next 单独保存。
adp 的 final_gamma 保留原版最后一次自适应更新后的定义。

状态、重试、结果回传
  bash STATUS.sh results/standard_mix_01
  bash PACK_RESULTS.sh results/standard_mix_01
  bash RETRY_FAILED.sh results/standard_mix_01
  bash STOP.sh results/standard_mix_01

PACK_RESULTS 可随时打包已完成试验，不必等待整批结束。
RETRY_FAILED 请在 array 全部退出后执行，只重提失败/未完成试验，旧尝试不覆盖。
STOP 只取消该 run-dir 关联的作业，不取消你的其他超算任务。
自动统计/打包作业使用 afterany 依赖，即使失败也能导出诊断信息。
结果 ZIP 位于 results/standard_mix_01/exports/，完成包带 complete，部分包带 partial。
将该 ZIP 上传给我即可分析。不是只看统计作业是否 COMPLETED；还要看 HANDOFF.json。

statistics/summary_mean_std.csv 为跨全部已完成试验的汇总；raw_results.csv 为原始记录，
coverage.csv 报告缺失次数，status.csv 报告每个任务状态，HANDOFF.json 包含完整性及
逐轮 gamma 检查。每个 unit/output 内的表只是单次试验，不能当成 10 次重复的总表。
未完成/失败项不补零。无 best-known cut 数据时 cut_ratio 保留 NaN。
原始 .mat、舍入分配、逐轮历史、日志、数据和代码快照均包含在结果 ZIP。

验证范围
已完成低秩/稀疏公式与直接小矩阵计算对照、原基线内核和原种子核对、
模拟 Slurm/MATLAB 的提交/失败重试/部分打包，以及对原始统计表的复算检查。
模拟数据未放进此包。制包环境没有 MATLAB 和 CU，未声称实际跑过超算实验。
计算节点会先执行公式自检，并在每个试验结束后检查混合更新的全部历史。
任务正常结束、公式检查通过，不等同于达到可行性、驻点或性能提升。
