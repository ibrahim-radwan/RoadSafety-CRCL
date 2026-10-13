# Third-party notices

This project's own code is licensed under the **GNU General Public License
v3.0** ([LICENSE](LICENSE)). The repository also contains code and binaries
from third parties. They remain under their own licenses, listed below with
what this project changed.

- The BSD-3-Clause and MIT components are GPL-compatible.
- **The radar firmware images (§2) are separate programs**, distributed
  alongside the code ("aggregation" in GPL terms). They are **not** covered by
  the GPL, and keep the TI terms below: TI devices only, unmodified binaries,
  no reverse engineering.

The TI license information comes from the TI package manifests on the
project's Windows firmware-build station (paths below). The manifests are not
copied into this repository.

---

## 1. RoboSense rs_driver: `lidar_standalone_capture/rs_driver/`

- **Source:** https://github.com/RoboSense-LiDAR/rs_driver, base commit
  `897b14d3bdb6186a75df27ba51b65b5bd5557723`
- **License:** 3-clause BSD, Copyright (c) 2020 RoboSense. The full text is in
  [LICENSE](lidar_standalone_capture/rs_driver/LICENSE).
- **Modified** for this project; every change is listed in
  [NOTICE.md](lidar_standalone_capture/rs_driver/NOTICE.md). RoboSense has not
  reviewed or endorsed these changes.

## 2. Radar firmware images: `cascaded_radar_capture_pipeline/firmware/`

The `.appimage`, `.rprc` and `.xer5f` files (and the `.map` linker maps) are
binaries built by this project from the TI mmWave MCU+ SDK cascade test
application, with this project's modifications (DSS-L3 memory init, waveform
variants). The firmware source lives in the project's separate Windows
repository. Besides this project's code, the images contain:

| Component | Version | License | Manifest (on the build station) |
|---|---|---|---|
| mmWave control (`libmmwave_cascade`), mmWaveLink (`libmmwavelink_cascade`), test logger | mmWave MCU+ SDK 04.04.00.01 | BSD-3-Clause, © Texas Instruments | `mmwave_mcuplus_sdk_04_04_00_01\docs\mmwave_mcuplus_sdk_software_manifest.html` |
| **AWR2243 firmware patch** (`xwr22xx_metaImage.h`, downloaded to the radar front ends at start-up) | mmWave DFP 02.02.04.00 | **TI Text File License** (the DFP manifest; the file header says BSD-3-Clause, and the stricter one is followed). See §5. | `mmwave_dfp_02_02_04_00\mmwave_dfp_manifest.html` |
| MCU+ SDK drivers, board support, DPL | MCU+ SDK AM273x 08.05.00 | BSD-3-Clause, © Texas Instruments | `mcu_plus_sdk_am273x_08_05_00_24\docs\mcu_plus_sdk_am273x_manifest.html` |
| FreeRTOS Kernel | 10.4.3 | MIT, © Amazon.com, Inc. or its affiliates | MCU+ SDK manifest |
| Tiny printf | 4.0.0 | MIT, © Marco Paland | MCU+ SDK manifest |
| TI Arm Clang runtime library (`libc.a`) | ti-cgt-armllvm 4.0.4 LTS | Apache-2.0 WITH LLVM-exception, MIT, BSD-style and public-domain parts (per its manifest) | `ccs2050\ccs\tools\compiler\ti-cgt-armllvm_4.0.4.LTS\TI_Arm_Clang_Compiler_4.0.3_manifest.html` |

**Conditions that follow for these images (because of the AWR2243 patch):**
- they may be redistributed **in binary form, unmodified**;
- they may be used **only with TI devices** (the AWR2243/AM273x radar);
- **no reverse engineering, decompilation or disassembly** of the patch;
- this notice and the licence texts in §4–§5 must accompany them.

## 3. Texas Instruments flashing tools: `flashing/boot/`

- **Included:**
  - UART flashing scripts: `uart_uniflash.py`, `uart_uniflash_gui.py`,
    `uart_bootloader.py`, `uart_bootloader_bufio.py`;
  - the SoC-ID parser (`socid_parser/`);
  - the prebuilt secondary bootloaders (`sbl_prebuilt/am273x-evm/*.tiimage`);
  - image-tool sources and scripts (`multicoreImageGen/`, `out2rprc/`,
    `xipGen/`).
- **Origin:** TI MCU+ SDK for AM273x 08.05.00.24 (`tools/boot/`).
- **License:** BSD-3-Clause, © Texas Instruments Incorporated. TI's BSD-3 text
  is in §4. The MCU+ SDK manifest (above) covers the SDK's boot tools. The
  build-station summary did not quote individual manifest entries for these
  specific files, and several scripts carry no header in this copy.
- **Not included:**
  - `signing/`: image-signing tools, including TI's development RSA keys;
  - the Windows `.exe` builds of the image tools.

  Neither is needed on the Jetson, which only flashes ready-made images.
- **This project's addition:** `flash_variant.sh`.

---

## 4. License texts

### BSD-3-Clause (Texas Instruments): as in TI's source headers

```text
Copyright (C) Texas Instruments Incorporated

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

  Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the
  distribution.

  Neither the name of Texas Instruments Incorporated nor the names of
  its contributors may be used to endorse or promote products derived
  from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

### MIT (FreeRTOS Kernel © Amazon.com, Inc. or its affiliates; Tiny printf © Marco Paland)

```text
Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

### TI Arm Clang runtime library

Apache-2.0 WITH LLVM-exception for the LLVM-derived parts. The LLVM exception
waives the attribution requirement for object code compiled into a product.
The remaining runtime parts are listed with their licenses in the compiler
manifest named in §2.

## 5. TI Text File License (AWR2243 firmware patch)

Verbatim from `mmwave_dfp_02_02_04_00\mmwave_dfp_manifest.html`:

```text
TEXAS INSTRUMENTS TEXT FILE LICENSE

Copyright (c) 2020 Texas Instruments Incorporated

All rights reserved not granted herein.

Limited License. 

Texas Instruments Incorporated grants a world-wide, royalty-free, non-exclusive license under copyrights and patents it now or hereafter owns or controls to make, have made, use, import, offer to sell and sell ("Utilize") this software subject to the terms herein. With respect to the foregoing patent license, such license is granted solely to the extent that any such patent is necessary to Utilize the software alone. The patent license shall not apply to any combinations which include this software, other than combinations with devices manufactured by or for TI ("TI Devices"). No hardware patent is licensed hereunder.

Redistributions must preserve existing copyright notices and reproduce this license (including the above copyright notice and the disclaimer and (if applicable) source code license limitations below) in the documentation and/or other materials provided with the distribution

Redistribution and use in binary form, without modification, are permitted provided that the following conditions are met:

* No reverse engineering, decompilation, or disassembly of this software is permitted with respect to any software provided in binary form.

* any redistribution and use are licensed by TI for use only with TI Devices.

* Nothing shall obligate TI to provide you with source code for the software licensed and provided to you in object code.

If software source code is provided to you, modification and redistribution of the source code are permitted provided that the following conditions are met:

* any redistribution and use of the source code, including any resulting derivative works, are licensed by TI for use only with TI Devices.

* any redistribution and use of any object code compiled from the source code and any resulting derivative works, are licensed by TI for use only with TI Devices.

Neither the name of Texas Instruments Incorporated nor the names of its suppliers may be used to endorse or promote products derived from this software without specific prior written permission.

DISCLAIMER.

THIS SOFTWARE IS PROVIDED BY TI AND TI"S LICENSORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL TI AND TI"S LICENSORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Not included (obtain from the vendors)

- Stereolabs ZED SDK and its `pyzed` wheel.
- NVIDIA's PyTorch wheel for JetPack.
- Vendor PDF documents (TI and RoboSense manuals).
- TI SDKs, the TI image-signing tools and keys.
