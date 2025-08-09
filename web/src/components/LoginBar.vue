<template>
  <el-menu :default-active="activeIndex" class="el-menu-demo" mode="horizontal" :ellipsis="false" router>
    <el-menu-item >
      <img :src="logo" alt="Logo" style="width: 200px">
    </el-menu-item>
    <div class="flex-grow" />
    <el-menu-item index="/user/account/login/">登录</el-menu-item>
    <el-menu-item index="/user/account/register/">注册</el-menu-item>

  </el-menu>
</template>

<script lang="ts" setup>
import { ref } from 'vue'
import logo from '@/assets/long.jpg';

const activeIndex = ref('/user/account/login/')

</script>
<style scoped>
.el-menu {
  height: 100%;
}

.flex-grow {
  flex-grow: 1;
}
</style>