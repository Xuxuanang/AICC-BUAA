<template>
    <div class="common-layout">
        <el-container class="container">
            <el-header class="header">
                <LoginBar></LoginBar>
            </el-header>
            <el-main class="main">
                <router-view class="user_account"></router-view>
            </el-main>
        </el-container>
    </div>
</template>
  
<script lang="ts" setup>
import LoginBar from '@/components/LoginBar.vue'

</script>
  
<style lang="scss" scoped>
.container {
    height: 100%; // 保证页面没有滚动条
}

.header {
    background: linear-gradient(180deg, #f5f4f6, #e6ebf7);
    height: 6vh;
    margin: 0;
    padding: 0;
}

.main {
    background: linear-gradient(180deg, #f5f4f6, #e6ebf7);
    height: 94vh;
    margin: 0;
    padding: 0;
}
</style>
  