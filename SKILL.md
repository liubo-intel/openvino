---
name: dump_ov_trace_log
description: 在openvino中添加trace log dump功能，方便调试和性能分析。可以通过环境变量控制是否开启trace log dump，开启后可以dump openvino中指定的node/函数执行时间。
---
# CPU Linux 平台上的trace log dump功能
1. 在CPU平台上，已经实现了trace log dump功能，可以通过环境变量控制是否开启trace log dump，开启后可以dump openvino中指定的node/函数执行时间。相关代码已经提交到当前分支，形成的patch为'CPU_Linux_trace_log_dump.patch'。
2. 使用方法:
运行时定义环境变量： export LINUX_PERF=dump
程序析构的时候 dump trace文件, 文件名为perf_dump.json
openvino中dump的node/函数为inference 运行时的node函数以及push/pull函数.

# GPU以及windows平台上的trace log dump功能
1. 参考CPU平台上的实现以及opencl_rag repo, 在GPU以及windows平台上添加了trace log dump功能。dump范围覆盖inference运行时的node执行函数，以及主要数据处理函数（如set_input_data、set_output_memory、set_arguments、execute）。
2. 开启方式（Linux/Windows通用）：
	- 推荐环境变量：`OV_GPU_TRACE_DUMP=1`
	- 输出目录可选：`OV_GPU_TRACE_DUMP_DIR=<dump_dir>`（默认当前目录）
3. 兼容opencl_rag/opencl_tag环境变量：
	- 若设置`CLI_ChromePerformanceTiming=1`，也会开启GPU trace dump
	- 若设置`CLI_DumpDir=<dump_dir>`，默认输出到该目录
	- 输出目录优先级（从高到低）：
	  - `OV_GPU_TRACE_DUMP_DIR`
	  - `CLI_DumpDir`
	  - 当前工作目录
	- 含义说明：
	  - `CLI_ChromePerformanceTiming`：控制“开不开trace”。`1`表示开启，`0`或不设置表示关闭。
	  - `CLI_ChromePerformanceTimingPerKernel`：控制“trace细不细（按不按每个kernel）”。`1`表示按每个kernel记录（更细、开销更高）；`0`表示不按每个kernel展开（更简洁、开销更低）。
4. Linux兼容CPU路径：
	- 若设置`LINUX_PERF=dump`，GPU trace dump也可被触发
5. 输出文件：
	- 默认文件名：`ov_gpu_trace_dump.json`
	- 文件名固定为`ov_gpu_trace_dump.json`
	- 输出格式：Chrome Trace Event JSON，可用`chrome://tracing`或Perfetto查看
6. 示例：
	- Linux:
	  ```bash
	  export OV_GPU_TRACE_DUMP=1
	  export OV_GPU_TRACE_DUMP_DIR=/tmp/ov_trace
	  ./your_app
	  ```
	- Windows PowerShell:
	  ```powershell
	  $env:OV_GPU_TRACE_DUMP = "1"
	  $env:OV_GPU_TRACE_DUMP_DIR = "C:\\trace_dir"
	  .\your_app.exe
	  ```

7. opencl intercept（opencl_tag/opencl_rag）使用方法（用于补充OpenCL层面的trace）
	- 目的：
	  - `OV_GPU_TRACE_DUMP`聚焦OpenVINO GPU插件内部流程（node/data_flow）。
	  - opencl intercept聚焦OpenCL API/kernel执行层。
	  - 两者可同时开启，用于关联“框架层”与“OpenCL层”耗时。
	- 编译opencl intercept库（以opencl_tag仓库为例）：
	  - 进入仓库并拉取子模块：
	    ```bash
      git clone https://github.com/tiger100256-hu/opencl_tag.git  //fork: https://github.com/liubo-intel/opencl_tag
	    cd opencl_tag
	    git submodule update --init --recursive
	    ```
	  - Linux构建：
	    ```bash
	    mkdir -p build && cd build
	    cmake .. -DCMAKE_BUILD_TYPE=Release
	    cmake --build . --parallel 8
	    ```
	  - Windows（PowerShell）构建：
	    ```powershell
	    cd C:\path\to\opencl_tag
	    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
	    cmake --build build --config Release --parallel
	    ```
	- 产物位置（常见）：
	  - Linux：`bin/intercept/libOpenCL.so`
	  - Windows：`bin/intercept/OpenCL.dll`
	- 替换策略（推荐优先级）：
	  - 首选“仅对当前应用生效”的方式，不替换系统OpenCL：
	    - Linux：通过`LD_PRELOAD`注入`libOpenCL.so`
	    - Windows：将`OpenCL.dll`拷贝到应用可执行文件同目录（优先于系统路径加载）
	  - 不建议直接覆盖系统驱动目录下的OpenCL运行时库。
	- Linux运行示例（推荐）：
	  ```bash
	  export CLI_ChromePerformanceTiming=1
	  export CLI_DumpDir=/tmp/ov_trace
	  export CLI_ChromePerformanceTimingPerKernel=0
	  export LD_PRELOAD=/home/liubo/my_project/opencl_tag/bin/intercept/libOpenCL.so
	
	  export OV_GPU_TRACE_DUMP=1
	  export OV_GPU_TRACE_DUMP_DIR=/tmp/ov_trace
	  ./your_app
	  ```
	- Windows运行示例（推荐）：
	  ```powershell
	  Copy-Item C:\path\to\opencl_tag\build\bin\intercept\OpenCL.dll .\ -Force
	
	  $env:CLI_ChromePerformanceTiming = "1"
	  $env:CLI_DumpDir = "C:\trace_dir"
	  $env:CLI_ChromePerformanceTimingPerKernel = "0"
	
	  $env:OV_GPU_TRACE_DUMP = "1"
	  $env:OV_GPU_TRACE_DUMP_DIR = "C:\trace_dir"
	  .\your_app.exe
	  ```
	- trace文件查看：
	  - OpenVINO GPU插件trace：`ov_gpu_trace_dump.json`
	  - opencl intercept trace：由`CLI_ChromePerformanceTiming`在`CLI_DumpDir`下生成（文件名以实际输出为准）
	  - 使用`chrome://tracing`或Perfetto同时加载多份json进行对照分析。
	- 常见问题排查：
	  - 未生成opencl trace：优先检查是否命中拦截库加载（Linux检查`LD_PRELOAD`路径，Windows确认dll与exe同目录）。
	  - 只看到OpenCL trace看不到OV trace：确认`OV_GPU_TRACE_DUMP=1`已设置。
	  - 路径权限问题：确保`CLI_DumpDir`与`OV_GPU_TRACE_DUMP_DIR`可写。
