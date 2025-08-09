<template>
    <el-container class="container">
        <div class="slogan">基于AST和KG的深度学习框架间模型代码迁移系统</div>

        <el-form label-width="54px" class="login-form">
            <el-form-item label="用户名">
                <el-input v-model="username" />
            </el-form-item>
            <el-form-item label="密码">
                <el-input v-model="password" show-password />
            </el-form-item>
            <el-form-item>
                <el-button type="primary" @click="login">登录</el-button>
            </el-form-item>
        </el-form>
    </el-container>
</template>

<script lang="ts" setup>
import { ref } from 'vue'
import $ from 'jquery'
import { useStore } from 'vuex'
import router from '../../../router/index'

const store = useStore();
const username = ref('');
const password = ref('');
const error_message = ref('');

const jwt_token = localStorage.getItem("jwt_token");
if (jwt_token) {
    store.commit("updateToken", jwt_token);
    store.dispatch("getinfo", {
        success() {
            router.push({ name: "home" });
        }
    })
}

const login = () => {
    error_message.value = "";
    store.dispatch("login", {
        username: username.value,
        password: password.value,
        success() {
            store.dispatch("getinfo", {  // 动态获取当前用户信息
                success() {
                    router.push({ name: 'home' });
                }
            })
        },
        error() {
            error_message.value = "用户名或密码错误";
        }
    })
}

</script>

<style scoped>
.container {
    background: radial-gradient(circle at 100% 0, #efe8eb 0, rgba(239, 232, 235, 0) 33%),
        radial-gradient(circle at 100% 25%, #e8ebea 0, hsla(160, 7%, 92%, 0) 39%),
        radial-gradient(circle at 100% 36%, hsla(160, 7%, 92%, .6) 0, hsla(160, 7%, 92%, 0) 38%),
        linear-gradient(180deg, #efeaef, #dcdcf5 99%);
    display: flex;
    flex-direction: column;
    /* 垂直方向排列子元素 */
    height: 100%;
}

.slogan {
    font-size: 40px;
    align-self: center;
    margin: 30px;
}

.login-form {
    align-self: center;
}

.el-button {
    width: 230px;
}
</style>