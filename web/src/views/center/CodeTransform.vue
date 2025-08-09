<template>
    <el-container class="container">

        <el-alert v-if="copyAlertVisible" title="复制成功!" type="success" center show-icon @close="copyAlertDisappear"
            class="copy-alert" />
        <el-alert v-if="downloadAlertVisible" title="文件已经保存在本地！" type="success" center show-icon @close="downloadAlertDisappear"
            class="copy-alert" />

        <el-row>
            <el-col :span="10" class="main-code">
                <div class="code-header">
                    <span class="code-lang">源代码</span>

                    <span class="parameter-setting hover-underline" @click="selectParameter = true, setParameter()">
                        <el-icon>
                            <Setting />
                        </el-icon>
                        <span class="parameter-setting-text">参数设置</span>
                    </span>

                    <el-dialog v-model="selectParameter" title="参数设置" width="30%" align-center class="parameter-setting-dialog">

                        <p class="select-title">源框架名称</p>
                        <el-select v-model="source_framework" class="m-2 parameter" placeholder="源框架名称" size="large">
                            <el-option v-for="  item   in   options_source_framework  " :key="item.value"
                                :label="item.label" :value="item.value" />
                        </el-select>

                        <p class="select-title">源框架版本</p>
                        <el-select v-model="source_version" class="m-2 parameter" placeholder="源框架版本" size="large">
                            <el-option v-for="  item   in   options_source_version  " :key="item.value"
                                :label="item.label" :value="item.value" />
                        </el-select>

                        <p class="select-title">目标框架名称</p>
                        <el-select v-model="target_framework" class="m-2 parameter" placeholder="目标框架名称" size="large">
                            <el-option v-for="  item   in   options_target_framework.filter(item => item.value !== source_framework) " :key="item.value"
                                :label="item.label" :value="item.value" />
                        </el-select>

                        <p class="select-title">目标框架版本</p>
                        <el-select v-model="target_version" class="m-2 parameter" placeholder="目标框架版本" size="large">
                            <el-option v-for="  item   in   options_target_version  " :key="item.value"
                                :label="item.label" :value="item.value" />
                        </el-select>

                        <el-divider border-style="dashed" />

                        <div class="mb-2 flex items-center text-sm code-option">
                            <el-radio-group v-model="code_from" class="ml-4">
                                <el-radio label="input" size="large">网页输入</el-radio>
                                <el-radio label="upload" size="large">上传文件</el-radio>
                                <el-radio label="project" size="large">选择项目</el-radio>
                            </el-radio-group>
                        </div>

                        <p class="select-title" v-show="code_from === 'input'">模型选择</p>
                        <el-select v-model="models" v-show="code_from === 'input'" class="m-2 parameter"
                            placeholder="模型选择" size="large">
                            <el-option v-for="  item   in   options_models  " :key="item.value" :label="item.label"
                                :value="item.value" />
                        </el-select>

                        <el-upload v-model:file-list="fileList" v-show=" code_from === 'upload' " class="upload-demo"
                            action="http://127.0.0.1:3000/transform/upload/" :data={page_id:curPageId} :limit=" 1 "
                            :on-exceed=" handleExceed ">
                            <el-button type="primary">点击上传文件</el-button>
                            <template #tip>
                                <div class="el-upload__tip">
                                    最多上传一个文件(文件类型为py)
                                </div>
                            </template>
                        </el-upload>

                        <template #footer>
                            <span class="dialog-footer">
                                <el-button @click="selectParameter = false">取消</el-button>
                                <el-button type="primary" @click="selectParameter = false, confirmParameter()">确认</el-button>
                            </span>
                        </template>
                    </el-dialog>

                    <span class="code-copy hover-underline" @click=" copySource ">
                        <el-icon>
                            <DocumentCopy />
                        </el-icon>
                        <span class="code-copy-text">复制代码</span>
                    </span>
                </div>
                <v-ace-editor v-model:value=" source_code " lang="python" theme="dracular" class="code-editor" />
            </el-col>

            <el-col :span=" 4 " class="main-conversion">
                <el-button class="conversion-button" type="primary" :icon=" Connection " size="large"
                    @click=" transform ">开始迁移</el-button>
            </el-col>

            <el-col :span=" 10 " class="main-code">
                <div class="code-header">
                    <span class="code-lang">目标代码</span>

                    <span class="download-code hover-underline" @click="downloadCode()" >
                        <el-icon><Download /></el-icon>
                        <span class="download-code-text">下载代码</span>
                    </span>

                    <span class="code-copy hover-underline" @click="copyTarget()" >
                        <el-icon><DocumentCopy /></el-icon>
                        <span class="code-copy-text">复制代码</span>
                    </span>
                </div>
                <v-ace-editor v-model:value=" target_code " lang="python" theme="dracular" class="code-editor" />
            </el-col>
        </el-row>
    </el-container>
</template>

<script lang="ts" setup>
import { ref, onMounted, onBeforeUnmount } from 'vue'
import $ from 'jquery'
import $bus from '@/bus/index'
import { ElMessage } from 'element-plus'
import type { UploadProps, UploadUserFile } from 'element-plus'
import { Connection, DocumentCopy, Setting, Download } from '@element-plus/icons-vue'
import { VAceEditor } from 'vue3-ace-editor'
import '../../../ace.config'
import { useStore } from 'vuex'

const store = useStore();

const curPageId = ref("1");  // 当前页面id

const copyAlertVisible = ref(false);  // 复制成功提示是否显示
const downloadAlertVisible = ref(false);
const selectParameter = ref(false)

const source_framework = ref('PyTorch')  // 源框架名称
const source_framework_true = ref('PyTorch')
const source_version = ref('1.5.0')  // 源框架版本
const source_version_true = ref('1.5.0')
const source_code = ref("Please input your code")  // 源代码字符串

const target_framework = ref('MindSpore')  // 目标框架名称
const target_framework_true = ref('MindSpore')
const target_version = ref('1.5.0')  // 目标框架版本
const target_version_true = ref('1.5.0')
const target_code = ref("Waiting...")  // 目标代码字符串

const code_from = ref('input')  // 代码来源：网页输入 或 上传文件
const code_from_true = ref('input')
const models = ref('Custom input')  // 网页输入的模型：自定义输入或具体模型如resnet
const models_true = ref('Custom input')

const options_models = [  // 网页输入的模型
    {
        value: 'Custom input',
        label: 'Custom input',
    },
    {
        value: 'ResNet50',
        label: 'ResNet50',
    },
]

const options_source_framework = [  // 源框架名称
    {
        value: 'PyTorch',
        label: 'PyTorch',
    },
    {
        value: 'PaddlePaddle',
        label: 'PaddlePaddle',
    },
    {
        value: 'MindSpore',
        label: 'MindSpore',
    },
]

const options_target_framework = [  // 目标框架名称
    {
        value: 'PyTorch',
        label: 'PyTorch',
    },
    {
        value: 'PaddlePaddle',
        label: 'PaddlePaddle',
    },
    {
        value: 'MindSpore',
        label: 'MindSpore',
    },
]

const options_source_version = [  // 源框架版本
    {
        value: '1.5.0',
        label: '1.5.0'
    },
]

const options_target_version = [  // 目标框架版本
    {
        value: '1.5.0',
        label: '1.5.0'
    },
]

const fileList = ref<UploadUserFile[]>([])  // 初始化空的上传文件列表

///////////////////////////////////////////////////、、、、、、、、、、、、、、、、、、、、、、、、、


onMounted(() => {
    loadPage();

    $bus.on('loadPageTransform', () => {
        loadPage();
    });

    // 保存当前对话框
    $bus.on('savePageTransform', () => {
        savePage();
    });
});

onBeforeUnmount(() => {
    savePage();
});

// 加载当前迁移对话
const loadPage = () => {
    if (store.state.page_id !== 0) {
        $.ajax({
            url: "http://127.0.0.1:3000/transform/page/info/",
            type: "post",
            headers: {
                Authorization: "Bearer " + store.state.user.token,
            },
            data: {
                page_id: store.state.page_id,
            },
            success(resp) {
                source_framework.value = source_framework_true.value = resp.sourceFramework;
                source_version.value = source_version_true.value = resp.sourceVersion;
                source_code.value = resp.sourceCode;
                target_framework.value = target_framework_true.value = resp.targetFramework;
                target_version.value = target_version_true.value = resp.targetVersion;
                target_code.value = resp.targetCode;

                code_from.value = code_from_true.value = resp.codeFrom;
                models.value = models_true.value = resp.models;
            }
        });
    }
};

// 保存当前对话
const savePage = () => {
    $.ajax({
            url: "http://127.0.0.1:3000/transform/page/save/",
            type: "post",
            headers: {
                Authorization: "Bearer " + store.state.user.token,
            },
            data: {
                page_id: store.state.page_id,
                source_framework: source_framework_true.value,
                source_version: source_version_true.value,
                target_framework: target_framework_true.value,
                target_version: target_version_true.value,
                code_from: code_from_true.value,
                models: models_true.value,
                source_code: source_code.value,
                target_code: target_code.value,
            }
        });
}

async function copyAlertDisappear() {
    copyAlertVisible.value = false;
}

async function downloadAlertDisappear() {
    downloadAlertVisible.value = false;
}

const handleExceed: UploadProps['onExceed'] = (files, uploadFiles) => {
    ElMessage.warning(
        `最多上传 1 个文件！`
    )
}

// 参数设置打开时
async function setParameter() {
    source_framework.value = source_framework_true.value;
    source_version.value = source_version_true.value;
    target_framework.value = target_framework_true.value;
    target_version.value = target_version_true.value;

    code_from.value = code_from_true.value;
    models.value = models_true.value;
}

// 参数设置确认
async function confirmParameter() {
    // 检查


    // 更新真值
    source_framework_true.value = source_framework.value;
    source_version_true.value = source_version.value;
    target_framework_true.value = target_framework.value;
    target_version_true.value = target_version.value;

    code_from_true.value = code_from.value;
    models_true.value = models.value;

    // 更新源码输入框
    if (code_from.value === 'input') {
        if (models.value === 'Custom input') {
            source_code.value = 'Please input your code';
        } else if (source_framework_true.value === 'PyTorch' && models.value === 'ResNet50') {
            source_code.value = "import numpy\n" +
                "import torch\n" +
                "from torch import nn\n" +
                "import torch.nn.functional as F\n" +
                "import sys\n" +
                "\n" +
                "\n" +
                "class Bottleneck(nn.Module):\n" +
                "    def __init__(self, in_channels, out_channels, stride=[1, 1, 1], padding=[0, 1, 0], first=False) -> None:\n" +
                "        super(Bottleneck, self).__init__()\n" +
                "        self.bottleneck = nn.Sequential(\n" +
                "            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], padding=padding[0], bias=False),\n" +
                "            nn.BatchNorm2d(out_channels),\n" +
                "            nn.ReLU(inplace=True),  # 原地替换 节省内存开销\n" +
                "            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=padding[1], bias=False),\n" +
                "            nn.BatchNorm2d(out_channels),\n" +
                "            nn.ReLU(inplace=True),  # 原地替换 节省内存开销\n" +
                "            nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, stride=stride[2], padding=padding[2], bias=False),\n" +
                "            nn.BatchNorm2d(out_channels * 4)\n" +
                "        )\n" +
                "\n" +
                "        # shortcut 部分\n" +
                "        # 由于存在维度不一致的情况 所以分情况\n" +
                "        self.shortcut = nn.Sequential()\n" +
                "        if first:\n" +
                "            self.shortcut = nn.Sequential(\n" +
                "                # 卷积核为1 进行升降维\n" +
                "                # 注意跳变时 都是stride==2的时候 也就是每次输出信道升维的时候\n" +
                "                nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, stride=stride[1], bias=False),\n" +
                "                nn.BatchNorm2d(out_channels * 4)\n" +
                "            )\n" +
                "\n" +
                "    def forward(self, x):\n" +
                "        out = self.bottleneck(x)\n" +
                "        out += self.shortcut(x)\n" +
                "        out = F.relu(out)\n" +
                "        return out\n" +
                "\n" +
                "\n" +
                "# 采用bn的网络中，卷积层的输出并不加偏置\n" +
                "class ResNet50(nn.Module):\n" +
                "    def __init__(self, Bottleneck, num_classes=10) -> None:\n" +
                "        super(ResNet50, self).__init__()\n" +
                "        self.in_channels = 64\n" +
                "        # 第一层作为单独的 因为没有残差快\n" +
                "        self.conv1 = nn.Sequential(\n" +
                "            # nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),\n" +
                "            nn.Conv2d(3, 64, 1, 1, 0, bias=False),\n" +
                "            nn.BatchNorm2d(64),\n" +
                "            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)\n" +
                "        )\n" +
                "\n" +
                "        # conv2\n" +
                "        self.conv2 = self._make_layer(Bottleneck, 64, [[1, 1, 1]] * 3, [[0, 1, 0]] * 3)\n" +
                "\n" +
                "        # conv3\n" +
                "        self.conv3 = self._make_layer(Bottleneck, 128, [[1, 2, 1]] + [[1, 1, 1]] * 3, [[0, 1, 0]] * 4)\n" +
                "\n" +
                "        # conv4\n" +
                "        self.conv4 = self._make_layer(Bottleneck, 256, [[1, 2, 1]] + [[1, 1, 1]] * 5, [[0, 1, 0]] * 6)\n" +
                "\n" +
                "        # conv5\n" +
                "        self.conv5 = self._make_layer(Bottleneck, 512, [[1, 2, 1]] + [[1, 1, 1]] * 2, [[0, 1, 0]] * 3)\n" +
                "\n" +
                "        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))\n" +
                "        self.avgpool = nn.MaxPool2d((7, 7))\n" +
                "        self.fc = nn.Linear(2048, num_classes)\n" +
                "\n" +
                "    def _make_layer(self, block, out_channels, strides, paddings):\n" +
                "        layers = []\n" +
                "        # 用来判断是否为每个block层的第一层\n" +
                "        flag = True\n" +
                "        for i in range(0, len(strides)):\n" +
                "            layers.append(block(self.in_channels, out_channels, strides[i], paddings[i], first=flag))\n" +
                "            flag = False\n" +
                "            self.in_channels = out_channels * 4\n" +
                "\n" +
                "        return nn.Sequential(*layers)\n" +
                "\n" +
                "    def forward(self, x):\n" +
                "        out = self.conv1(x)\n" +
                "        # print(out)\n" +
                "        # sys.exit()\n" +
                "        out = self.conv2(out)\n" +
                "        out = self.conv3(out)\n" +
                "        out = self.conv4(out)\n" +
                "        out = self.conv5(out)\n" +
                "        # out = self.avgpool(out)\n" +
                "\n" +
                "        out = out.mean(-1).mean(-1)\n" +
                "        out = out.reshape(x.shape[0], -1)\n" +
                "        out = self.fc(out)\n" +
                "        return out\n" +
                "\n" +
                "\n" +
                "class Resnet50_model(nn.Module):\n" +
                "    def __init__(self, args):\n" +
                "        super(Resnet50_model, self).__init__()\n" +
                "        self.model = ResNet50(Bottleneck, num_classes=args.num_classes)\n" +
                "\n" +
                "    def forward(self, x):\n" +
                "        return self.model(x)\n";
        } else if (source_framework_true.value === 'MindSpore' && models.value === 'ResNet50') {
            source_code.value = "import mindspore\n" +
                "\n" +
                "class Bottleneck(mindspore.nn.Cell):\n" +
                "\n" +
                "    def __init__(self, in_channels, out_channels, stride=[1, 1, 1], padding=[0, 1, 0], first=False) -> None:\n" +
                "        super(Bottleneck, self).__init__()\n" +
                "        self.bottleneck = mindspore.nn.SequentialCell(\n" +
                "            mindspore.nn.Conv2d(in_channels, out_channels, 1, stride[0], pad_mode='pad', padding=padding[0], has_bias=False),\n" +
                "            mindspore.nn.BatchNorm2d(out_channels), mindspore.nn.ReLU(), mindspore.nn.Conv2d(out_channels, out_channels, 3, stride[1], pad_mode='pad', padding=padding[1], has_bias=False), mindspore.nn.BatchNorm2d(out_channels), mindspore.nn.ReLU(), mindspore.nn.Conv2d(out_channels, out_channels * 4, 1, stride[2], pad_mode='pad', padding=padding[2], has_bias=False), mindspore.nn.BatchNorm2d(out_channels * 4))\n" +
                "        self.shortcut = mindspore.nn.SequentialCell()\n" +
                "        if first:\n" +
                "            self.shortcut = mindspore.nn.SequentialCell(mindspore.nn.Conv2d(in_channels, out_channels * 4, 1, stride[1], pad_mode='pad', has_bias=False), mindspore.nn.BatchNorm2d(out_channels * 4))\n" +
                "\n" +
                "    def construct(self, x):\n" +
                "        out = self.bottleneck(x)\n" +
                "        out += self.shortcut(x)\n" +
                "        out = mindspore.ops.ReLU()(out)\n" +
                "        return out\n" +
                "\n" +
                "class ResNet50(mindspore.nn.Cell):\n" +
                "\n" +
                "    def __init__(self, Bottleneck, num_classes=10) -> None:\n" +
                "        super(ResNet50, self).__init__()\n" +
                "        self.in_channels = 64\n" +
                "        self.conv1 = mindspore.nn.SequentialCell(mindspore.nn.Conv2d(3, 64, 1, 1, pad_mode='valid', padding=0, has_bias=False),\n" +
                "                                                 mindspore.nn.BatchNorm2d(64),\n" +
                "                                                 mindspore.nn.MaxPool2d(3, 2, 'same'))\n" +
                "        self.conv2 = self._make_layer(Bottleneck, 64, [[1, 1, 1]] * 3, [[0, 1, 0]] * 3)\n" +
                "        self.conv3 = self._make_layer(Bottleneck, 128, [[1, 2, 1]] + [[1, 1, 1]] * 3, [[0, 1, 0]] * 4)\n" +
                "        self.conv4 = self._make_layer(Bottleneck, 256, [[1, 2, 1]] + [[1, 1, 1]] * 5, [[0, 1, 0]] * 6)\n" +
                "        self.conv5 = self._make_layer(Bottleneck, 512, [[1, 2, 1]] + [[1, 1, 1]] * 2, [[0, 1, 0]] * 3)\n" +
                "        self.avgpool = mindspore.nn.MaxPool2d((7, 7), pad_mode='valid')\n" +
                "        self.fc = mindspore.nn.Dense(2048, num_classes)\n" +
                "\n" +
                "    def _make_layer(self, block, out_channels, strides, paddings):\n" +
                "        layers = []\n" +
                "        flag = True\n" +
                "        for i in range(0, len(strides)):\n" +
                "            layers.append(block(self.in_channels, out_channels, strides[i], paddings[i], first=flag))\n" +
                "            flag = False\n" +
                "            self.in_channels = out_channels * 4\n" +
                "        return mindspore.nn.SequentialCell(*layers)\n" +
                "\n" +
                "    def construct(self, x):\n" +
                "        out = self.conv1(x)\n" +
                "        out = self.conv2(out)\n" +
                "        out = self.conv3(out)\n" +
                "        out = self.conv4(out)\n" +
                "        out = self.conv5(out)\n" +
                "        out = out.mean(-1).mean(-1)\n" +
                "        out = out.reshape(x.shape[0], -1)\n" +
                "        out = self.fc(out)\n" +
                "        return out\n" +
                "\n" +
                "class Resnet50_model(mindspore.nn.Cell):\n" +
                "\n" +
                "    def __init__(self, args):\n" +
                "        super(Resnet50_model, self).__init__()\n" +
                "        self.model = ResNet50(Bottleneck, num_classes=args.num_classes)\n" +
                "\n" +
                "    def construct(self, x):\n" +
                "        return self.model(x)"
        }
    } else if (code_from.value === 'upload') {
        source_code.value = 'The file has been uploaded to the backend server';
    } else if (code_from.value === 'project') {
        source_code.value = 'please input your project';
        target_code.value = 'loading...';
    }
}

// 复制代码
async function copySource() {
    try {
        await navigator.clipboard.writeText(source_code.value);
        copyAlertVisible.value = true;
    } catch (err) {
        console.error('无法复制文本: ', err);
    }
}

async function copyTarget() {
    try {
        await navigator.clipboard.writeText(target_code.value);
        copyAlertVisible.value = true;
    } catch (err) {
        console.error('无法复制文本: ', err);
    }
}

async function downloadCode() {
    downloadAlertVisible.value = true;
}

// 开始迁移
async function transform() {
if (code_from_true.value === 'input') {
    $.ajax({
        url: "http://127.0.0.1:3000/transform/target/input/",
        type: "post",
        headers: {
                Authorization: "Bearer " + store.state.user.token,
            },
        data: {
            page_id: store.state.page_id,
            source_framework: source_framework_true.value,
            source_code: source_code.value,
            target_framework: target_framework_true.value
        },
        success(resp) {
            target_code.value = resp.target_code;
        }
    })
} else if (code_from_true.value === 'upload') {
    $.ajax({
        url: "http://127.0.0.1:3000/transform/target/upload/file/",
        type: "post",
        headers: {
                Authorization: "Bearer " + store.state.user.token,
            },
        data: {
            page_id: store.state.page_id,
            source_framework: source_framework_true.value,
            target_framework: target_framework_true.value
        },
        success(resp) {
            target_code.value = resp.target_code;
        }
    })
} else if (code_from_true.value === 'project') {
    $.ajax({
        url: "http://127.0.0.1:3000/transform/target/project/",
        type: "post",
        headers: {
                Authorization: "Bearer " + store.state.user.token,
            },
        data: {
            page_id: store.state.page_id,
            source_framework: source_framework_true.value,
            target_framework: target_framework_true.value
        },
        success(resp) {
            target_code.value = resp.target_code;
        }
    })
}
}

</script>

<style scoped>
.container {
    height: 100%;
    display: flex;
}

/* 复制成功提示 */
.copy-alert {
    width: 15%;
    /* 水平居中 */
    position: fixed;
    left: 0;
    right: 0;
    margin: 0 auto;
    /* 为了能点击到× */
    z-index: 1010;
}

.el-row {
    flex-grow: 1;
}

/* 代码部分的el-col */
.main-code {
    display: flex;
    flex-direction: column;
    justify-content: center;
    /* 水平居中 */
    align-items: center;
    /* 垂直居中 */
}

/* 代码编辑器头 */
.code-header {
    display: flex;
    flex-direction: row;
    align-items: center;
    background: #e3e8f6;
    border-radius: 7px 7px 0 0;
    padding: 0;
    margin-bottom: 0;
    height: 3%;
    width: 90%;
}

/* 代码编辑器名称 */
.code-lang {
    color: #120649;
    flex: 1 0 auto;
    font-size: 16px;
    font-weight: 600;
    letter-spacing: 0;
    padding-left: 14px;
    text-align: justify;
}

/* 参数设置 */
.parameter-setting {
    align-items: center;
    color: #7886a4;
    display: flex;
    font-family: PingFangSC-Regular;
    font-size: 13px;
    font-weight: 400;
    letter-spacing: 0;
    line-height: 14px;
    text-align: justify;
    user-select: none;
}

/* 参数设置的文本 */
.parameter-setting-text {
    margin-left: 7px;
    margin-right: 20px;
}

/* 参数设置的对话框 */
.parameter-setting-dialog {
    display: flex;
    flex-direction: column;
}

/* 参数设置的参数 */
.parameter {
    margin: 5px;
}

.select-title{
    margin:5px;
}

/* 参数设置的网页输入或上传文件选择 */
.code-option {
    margin: 10px;
}

/* 参数设置的上传文件 */
.upload-demo {
    margin: 10px;
}

/* 参数设置对话框的确认取消 */
.dialog-footer button:first-child {
    margin-right: 10px;
}

.download-code {
    align-items: center;
    color: #7886a4;
    display: flex;
    font-family: PingFangSC-Regular;
    font-size: 13px;
    font-weight: 400;
    letter-spacing: 0;
    line-height: 14px;
    text-align: justify;
    user-select: none;
}

.download-code-text {
    margin-left: 7px;
    margin-right: 20px;
}

/* 复制代码 */
.code-copy {
    align-items: center;
    color: #7886a4;
    display: flex;
    font-family: PingFangSC-Regular;
    font-size: 13px;
    font-weight: 400;
    letter-spacing: 0;
    line-height: 14px;
    text-align: justify;
    user-select: none;
}

/* 复制代码的文本 */
.code-copy-text {
    margin-left: 7px;
    margin-right: 20px;
}

/* 显示下划线 */
.hover-underline:hover {
    text-decoration: underline;
    color: blue;
}

/* 代码编辑器 */
.code-editor {
    height: 80%;
    width: 90%;
}

/* 转换部分的el-col */
.main-conversion {
    display: flex;
    justify-content: center;
    /* 水平居中 */
    align-items: center;
    /* 垂直居中 */
    height: 100%;
    /* 设置父元素的高度，确保垂直居中有效 */
}
</style>