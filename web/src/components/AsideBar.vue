<template>
    <el-row class="tac">
        <el-col :span="24">
            <el-button type="primary" size="large" class="new-transform" @click="addPage">新建迁移</el-button>

            <el-divider />
            <el-menu class="el-menu-vertical-demo">
                <el-menu-item v-for="page in pages" :key="page.id" :id="page.id" :index="page.id + ''"
                    v-bind:class="{ active: page.id === store.state.page_id }" @click="changePage(page.id)">
                    <span>
                        <el-icon>
                            <Menu />
                        </el-icon>
                    </span>
                    <span> {{ page.title }}</span>
                    <span class="edit-page" @click="editDialogVisible = true">
                        <el-icon class="hover">
                            <Edit />
                        </el-icon>
                    </span>
                    <span class="close-page" @click="closeDialogVisible = true">
                        <el-icon class="hover">
                            <CloseBold />
                        </el-icon>
                    </span>
                </el-menu-item>
            </el-menu>

            <el-dialog v-model="editDialogVisible" title="修改标题" width="30%">
                <el-input v-model="title_input" placeholder="请输入新标题" />
                <template #footer>
                    <span class="dialog-footer">
                        <el-button @click="editDialogVisible = false">取消</el-button>
                        <el-button type="primary" @click="editDialogVisible = false; editPage()">
                            确认
                        </el-button>
                    </span>
                </template>
            </el-dialog>

            <el-dialog v-model="closeDialogVisible" title="删除后无法恢复，是否继续删除？" width="30%">
                <template #footer>
                    <span class="dialog-footer">
                        <el-button @click="closeDialogVisible = false">取消</el-button>
                        <el-button type="danger" @click="closeDialogVisible = false; closePage()">
                            删除
                        </el-button>
                    </span>
                </template>
            </el-dialog>
        </el-col>
    </el-row>
</template>
  
<script lang="ts" setup>
import $ from 'jquery'
import { ref, onMounted } from 'vue'
import { useStore } from 'vuex'
import { useRoute } from 'vue-router'
import { Menu, Edit, CloseBold } from '@element-plus/icons-vue'
import $bus from '@/bus/index'

interface Page {
    id: number;
    title: string;
}

const store = useStore();
const route = useRoute();
const pages = ref<Page[]>([]);  // 迁移对话数组
const editDialogVisible = ref(false)
const closeDialogVisible = ref(false)
const title_input = ref('')

//////////////////////////////////////////////////////////////////////////

// 挂载完调用，组件的DOM已经生成
onMounted(() => {
    loadPages();
});

// 加载历史迁移对话列表
const loadPages = () => {
    $.ajax({
        url: "http://127.0.0.1:3000/transform/page/list/",
        type: "post",
        headers: {
            Authorization: "Bearer " + store.state.user.token,
        },
        data: {
            user_id: store.state.user.id
        },
        success(resp) {
            pages.value = [];
            for (var i = 0; i < resp.length; i++) {
                pages.value.push({
                    id: resp[i].id,
                    title: resp[i].title
                });
            }
            store.dispatch("updatePageId", {  // 更新当前对话id
                page_id: pages.value[0].id
            })
            loadPage();
        }
    });
}

// 加载当前迁移对话
const loadPage = () => {
    if (route.name === 'transform') {
        $bus.emit('loadPageTransform');
    } else if (route.name === 'detail') {
        $bus.emit('loadPageDetail');
    }
}

// 保存当前对话
const savePage = () => {
    if (route.name === 'transform') {
        $bus.emit('savePageTransform');
    }
}

// 新建迁移对话
const addPage = () => {
    savePage();  // 保存上一迁移对话

    $.ajax({
        url: "http://127.0.0.1:3000/transform/page/add/",
        type: "post",
        headers: {
            Authorization: "Bearer " + store.state.user.token,
        },
        data: {
            user_id: store.state.user.id
        },
        success(resp) {
            pages.value.push({
                id: resp.id,
                title: resp.title
            })
            store.dispatch("updatePageId", {  // 更新当前对话id
                page_id: resp.id
            })

            loadPage();
        }
    });
}

// 切换至另一对话框
const changePage = (id: number) => {
    savePage();  // 保存上一迁移对话

    store.dispatch("updatePageId", {  // 更新当前对话id
        page_id: id
    })

    loadPage();
}

const editPage = () => {
    $.ajax({
        url: "http://127.0.0.1:3000/transform/page/edit/",
        type: "post",
        headers: {
            Authorization: "Bearer " + store.state.user.token,
        },
        data: {
            page_id: store.state.page_id,
            title: title_input.value,
        },
        success(resp) {
            for (var i = 0; i < pages.value.length; i++) {
                if (pages.value[i].id === store.state.page_id) {
                    pages.value[i].title = title_input.value;
                }
            }
        }
    });
}

const closePage = () => {
    if (pages.value.length > 1) {
        $.ajax({
            url: "http://127.0.0.1:3000/transform/page/close/",
            type: "post",
            headers: {
                Authorization: "Bearer " + store.state.user.token,
            },
            data: {
                page_id: store.state.page_id,
            },
            success(resp) {
                loadPages();
            }
        });
    }
}
</script>

<style scoped>
.el-menu {
    background-color: transparent;
}

.active {
    color: #409eff;
}

.new-transform {
    margin-top: 30px;
    margin-bottom: 10px;
    margin-left: 22px;
    background-color: transparent;
    border-radius: 6px;
    color: #4955f5;
    display: flex;
    font-size: 16px;
}

.el-menu-item {
    margin: 0;
    padding: 0;
}

.el-icon {
    margin: 0;
    padding: 0;
}

.hover:hover {
    text-decoration: underline;
    background-color: rgba(109, 118, 246, 0.5);
}
</style>
  