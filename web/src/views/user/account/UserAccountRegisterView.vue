<template>
    <el-container class="container">
        <div class="slogan">基于AST和KG的深度学习框架间模型代码迁移系统</div>

        <el-form label-width="68px" class="login-form">
            <el-form-item label="用户名">
                <el-input v-model="username" />
            </el-form-item>
            <el-form-item label="密码">
                <el-input v-model="password" show-password />
            </el-form-item>
            <el-form-item label="确认密码">
                <el-input v-model="confirmedPassword" show-password />
            </el-form-item>
            <el-form-item>
                <div class="error-message">{{ error_message }}</div>
            </el-form-item>
            <el-form-item>
                <el-button type="primary" @click="register">注册</el-button>
            </el-form-item>
        </el-form>
    </el-container>
</template>

<script lang="ts" setup>
import { ref } from 'vue'
import router from '../../../router/index'
import $ from 'jquery'

const username = ref('');
const password = ref('');
const confirmedPassword = ref('')
const error_message = ref('');

const register = () => {
    $.ajax({
        url: "http://127.0.0.1:3000/user/account/register/",
        type: "post",
        data: {
            username: username.value,
            password: password.value,
            confirmedPassword: confirmedPassword.value,
        },
        success(resp) {
            if (resp.error_message === "success") {
                router.push({ name: "login" });
            } else {
                error_message.value = resp.error_message;
            }
        },
    });
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

div.error-message {
    color: red;
}

</style>