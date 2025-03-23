# AXCL 安装

## Raspberry Pi 5

### 准备事项

:::{Warning}
根据树莓派硬件批次不同，可能需要更新一下树莓派的 EEPROM 设置。具体步骤如下：
:::

如同 PC 中的 BIOS，EEPROM 设置独立于烧录 OS 的 TF 卡，烧录最新的树莓派镜像或者切换镜像版本并不会主动更新 EEPROM 的设置。首先执行 update：

```bash
sudo apt update && sudo apt full-upgrade
```

然后检查一下 EEPROM 中的版本：

```bash
sudo rpi-eeprom-update
```

如果看到的日期早于 `2023 年 12 月 6 日`，运行以下命令以打开 Raspberry Pi 配置 CLI：

```bash
sudo raspi-config
```

在 Advanced Options > Bootloader Version （引导加载程序版本） 下，选择 Latest （最新）。然后，使用 Finish 或 ESC 键退出 raspi-config。

执行以下命令，将固件更新到最新版本。

```bash
sudo rpi-eeprom-update -a
```

最后使用 `sudo reboot` 重新启动。重启后就完成了 EEPROM 中 firmware 的更新。


:::{Warning}
取决于使用的树莓派 kernel 状态，目前的修改是以 2024年11月18日 以前的树莓派刚烧录好的系统为例进行说明的，客户需要根据树莓派系统更新情况识别这个步骤是否必须。
:::

在当前的树莓派 kernel 和 M.2 HAT+ 组合中，可能会遇到如下限制：

> - PCIE Device 无法识别
> - PCIE MSI IRQ 无法申请多个

这些问题将导致安装失败或者子卡起不来。需要检查 Raspberry Pi 5 `/boot/firmware/config.txt` 文件，并进行修改。

如果是第三方的兼容 M.2 HAT+ 产品，需要注意供电问题；在 config.txt 中添加如下描述：

```bash
dtparam=pciex1
```
该描述可以默认打开 PCIE 功能；然后继续增加 PCIE 的设备描述：

```bash
dtoverlay=pciex1-compat-pi5,no-mip
```

完成修改并重启后，可以使用 `lspci` 命令检查加速卡是否正确被识别：

```bash
axera@raspberrypi:~ $ lspci
0000:00:00.0 PCI bridge: Broadcom Inc. and subsidiaries BCM2712 PCIe Bridge (rev 21)
0000:01:00.0 Multimedia video controller: Axera Semiconductor Co., Ltd Device 0650 (rev 01)
0001:00:00.0 PCI bridge: Broadcom Inc. and subsidiaries BCM2712 PCIe Bridge (rev 21)
0001:01:00.0 Ethernet controller: Raspberry Pi Ltd RP1 PCIe 2.0 South Bridge
```

其中 `Multimedia video controller: Axera Semiconductor Co., Ltd Device 0650 (rev 01)` 就是 AX650 加速卡。

### 软件包获取

:::{Warning}
请根据板卡上实际内存大小，选择对应版本的 AXCL 驱动。不同板卡厂家的 AXCL 驱动包原则上不能混用。
:::

- [芯茧配套](https://huggingface.co/AXERA-TECH/AXCL/tree/main)
- [HAT AI Module 配套]()

### 安装安装包

:::{Warning}
开发板需要编译支持，依赖 gcc, make, patch, linux-header-$(uname -r) 这几个包。需要提前安装好，或者保证安装时网络可用。
:::

将 aarch64 deb 包复制到树莓派开发板上，运行安装命令：

```bash
sudo dpkg -i axcl_host_aarch64_V2.16.1_20241118020146_NO4446.deb
```

安装将很快完成。安装时会自动增加环境变量，使得安装的 .so 和可执行程序可用。需要注意的是，如果需要可执行程序立即可用，还需要更新 bash 终端的环境：

```bash
source /etc/profile
```

如果是 ssh 的方式远程连接的板卡，还可以选择重连 ssh 进行自动更新(本机终端登录还可以重开一个终端进行自动更新)。

### 启动子卡

> - deb 包安装完成后将会自动启动子卡，无需再执行 `axclboot` 启动子卡
> - 主机启动时，将会自动启动子卡

### 安装成功测试

#### AXCL-SMI

正确安装 AXCL 驱动包后，AXCL-SMI 即安装成功，直接执行`axcl-smi`显示内容如下：

```bash
axera@raspberrypi:~ $ axcl-smi
+------------------------------------------------------------------------------------------------+
| AXCL-SMI  V3.0.1_20250318020150                                  Driver  V3.0.1_20250318020150 |
+-----------------------------------------+--------------+---------------------------------------+
| Card  Name                     Firmware | Bus-Id       |                          Memory-Usage |
| Fan   Temp                Pwr:Usage/Cap | CPU      NPU |                             CMM-Usage |
|=========================================+==============+=======================================|
|    0  AX650N                     V3.0.1 | 0000:01:00.0 |                147 MiB /      945 MiB |
|   --   56C                      -- / -- | 0%        0% |                 18 MiB /     7040 MiB |
+-----------------------------------------+--------------+---------------------------------------+

+------------------------------------------------------------------------------------------------+
| Processes:                                                                                     |
| Card      PID  Process Name                                                   NPU Memory Usage |
|================================================================================================|
```

#### UT TEST

正确安装 AXCL 驱动包后，默认会安装一系列 `UT Test` 执行程序。

```base
axera@raspberrypi:~ $ axcl_ut_
axcl_ut_cmm         axcl_ut_npu         axcl_ut_pool        axcl_ut_rt_engine   axcl_ut_rt_memory   axcl_ut_vdec
axcl_ut_ive         axcl_ut_package     axcl_ut_rt_context  axcl_ut_rt_init     axcl_ut_rt_p2p      axcl_ut_venc
axcl_ut_msys        axcl_ut_pcie_rc     axcl_ut_rt_device   axcl_ut_rt_latency  axcl_ut_rt_stream
```

可选择性运行 `axcl_ut_rt_engine`，`axcl_ut_rt_memory` 等用例进行自检。

```bash
axera@raspberrypi:~ $ axcl_ut_rt_engine
device index: 0, bus number: 1
[==========] Running 11 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 11 tests from axclrtDeviceTest
[ RUN      ] axclrtDeviceTest.Case01_AXCL_EngineInitFinal
[       OK ] axclrtDeviceTest.Case01_AXCL_EngineInitFinal (211 ms)
[ RUN      ] axclrtDeviceTest.Case02_axclrtEngineGetVNpuKind
[       OK ] axclrtDeviceTest.Case02_axclrtEngineGetVNpuKind (831 ms)
[ RUN      ] axclrtDeviceTest.Case03_axclrtEngineGetModelTypeFromMem
[       OK ] axclrtDeviceTest.Case03_axclrtEngineGetModelTypeFromMem (935 ms)
[ RUN      ] axclrtDeviceTest.Case04_axclrtEngineLoadFromMem
[       OK ] axclrtDeviceTest.Case04_axclrtEngineLoadFromMem (832 ms)
[ RUN      ] axclrtDeviceTest.Case05_axclrtEngineCreateContext
[       OK ] axclrtDeviceTest.Case05_axclrtEngineCreateContext (831 ms)
[ RUN      ] axclrtDeviceTest.Case06_axclrtEngineGetModelTypeFromModelId
[       OK ] axclrtDeviceTest.Case06_axclrtEngineGetModelTypeFromModelId (830 ms)
[ RUN      ] axclrtDeviceTest.Case07_axclrtEngineGetModelCompilerVersion
[       OK ] axclrtDeviceTest.Case07_axclrtEngineGetModelCompilerVersion (831 ms)
[ RUN      ] axclrtDeviceTest.Case08_axclrtEngineGetUsageFromModelId
[       OK ] axclrtDeviceTest.Case08_axclrtEngineGetUsageFromModelId (831 ms)
[ RUN      ] axclrtDeviceTest.Case09_axclrtEngineSetGetAffinity
[       OK ] axclrtDeviceTest.Case09_axclrtEngineSetGetAffinity (207 ms)
[ RUN      ] axclrtDeviceTest.Case10_axclrtEngineGetIOInfo
[       OK ] axclrtDeviceTest.Case10_axclrtEngineGetIOInfo (207 ms)
[ RUN      ] axclrtDeviceTest.Case11_axclrtEngineExecute
[       OK ] axclrtDeviceTest.Case11_axclrtEngineExecute (279 ms)
[----------] 11 tests from axclrtDeviceTest (6834 ms total)

[----------] Global test environment tear-down
[==========] 11 tests from 1 test suite ran. (6835 ms total)
[  PASSED  ] 11 tests.
============= UT PASS =============
```

### 卸载安装

```bash
sudo dpkg -r axclhost
```

:::{Warning}
卸载的包名不是安装包的名字，是项目包名，即 axclhost。
:::

deb 包卸载时，会自动 reset 子卡，子卡进入pcie download mode。
