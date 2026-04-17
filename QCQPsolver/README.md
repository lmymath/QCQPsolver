# QCQPsolver

这是一个最小化二次约束二次规划（QCQP）问题的示例项目，使用本地 Eigen 头文件进行矩阵计算。

## 项目结构

- `CMakeLists.txt`：项目根 CMake 配置
- `src/qcqp.cpp`：主程序入口，生成测试问题并求解
- `include/Eigen/`：本地 Eigen 头文件库

## 依赖

- CMake 3.15 或更高
- 支持 C++17 的编译器
- 无需额外外部库，Eigen 已随项目一并提供

## 编译

建议在项目根目录下新建 `build` 目录并从其中构建。

### Windows（PowerShell）

```powershell
cd d:\code\c++\QCQPsolver\QCQPsolver
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

如果使用 MinGW：

```powershell
cmake .. -G "MinGW Makefiles"
cmake --build .
```

如果使用 Visual Studio：

```powershell
cmake .. -G "Visual Studio 17 2022"
cmake --build . --config Release
```

### Linux / WSL

```bash
cd /path/to/QCQPsolver/QCQPsolver
mkdir -p build
cd build
cmake ..
cmake --build .
```

## 运行

构建完成后，执行生成的可执行文件：

### PowerShell / Windows

```powershell
.	cqp.exe
```

### Linux / WSL

```bash
./qcqp
```

该程序会生成一个随机 QCQP 问题并打印求解进度信息，包括可行性检查、对偶目标迭代信息和最终解结果。

## 常见问题

- 如果出现 `Eigen` 头文件找不到的错误，请确认 `include/Eigen` 目录存在，并且你在项目根目录下运行了 `cmake ..`。
- 如果使用 Visual Studio，请使用 `--config Release` 或 `--config Debug` 指定构建配置。

## 备注

该项目是一个单文件示例，适合作为 QCQP 算法和 Eigen 数值运算的测试基准。