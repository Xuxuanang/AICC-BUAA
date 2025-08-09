<template>
    <el-container class="container">
        <div class="code-header">
            <span class="code-lang">源抽象语法树</span>
            <span class="code-copy">
                <el-icon>
                    <DocumentCopy />
                </el-icon>
                <span class="code-copy-text">复制代码</span>
            </span>
        </div>
        <v-ace-editor v-model:value="source_tree" lang="python" theme="dracular" class="code-editor" />
    </el-container>

    <el-container class="container">
        <div class="code-header">
            <span class="code-lang">新抽象语法树</span>
            <span class="code-copy">
                <el-icon>
                    <DocumentCopy />
                </el-icon>
                <span class="code-copy-text">复制代码</span>
            </span>
        </div>
        <v-ace-editor v-model:value="target_tree" lang="python" theme="dracular" class="code-editor" />
    </el-container>
</template>

<script lang="ts" setup>
import { ref, onMounted } from 'vue'
import $ from 'jquery'
import $bus from '@/bus/index'
import { useStore } from 'vuex'
import { DocumentCopy } from '@element-plus/icons-vue'
import { VAceEditor } from 'vue3-ace-editor'
import '../../../ace.config'

const store = useStore();
const source_tree = ref('');
const target_tree = ref('');


///////////////////////////////////////////////////////////////////

onMounted(() => {
    loadPage();

    // 加载当前迁移对话
    $bus.on('loadPageDetail', () => {
        loadPage();
    });
});

// 加载当前迁移对话
const loadPage = () => {
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
            source_tree.value = resp.sourceTree;
            target_tree.value = resp.targetTree;
        }
    });
};
</script>

<style scoped>
.container {
    height: 50%;
    flex-direction: column;
    align-items: center;
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
    height: 8%;
    width: 60%;
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
    height: 60%;
    width: 60%;
}
</style>