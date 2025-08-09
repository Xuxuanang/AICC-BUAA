<template>
  <el-menu :default-active="activeIndex" class="el-menu-demo" mode="horizontal" :ellipsis="false" router>
    <el-menu-item index="/">
      <span style="font-size: 25px">基于AST和KG的深度学习框架间模型代码迁移系统
      </span>
    </el-menu-item>
    <div class="flex-grow" />
    <el-menu-item index="/center/transform/">迁移中心</el-menu-item>
    <el-menu-item index="/center/detail/">迁移细节</el-menu-item>

    <el-sub-menu v-if="$store.state.user.is_login" index="4">
      <template #title>{{ $store.state.user.username }}</template>
      <el-menu-item index="/user/account/login/" @click="logout">
          <span style="color:#409EFF">退出登录</span>
      </el-menu-item>

    </el-sub-menu>
  </el-menu>
</template>

<script lang="ts" setup>
import { ref } from 'vue'
import { useStore } from 'vuex'
import router from '../router/index'

const store = useStore();
const activeIndex = ref('/center/transform/')

const logout = () => {
  store.dispatch("logout");
}


</script>

<style scoped>
.el-menu {
  background: linear-gradient(180deg, #f5f4f6, #e6ebf7);
  height: 100%;
}

.flex-grow {
  flex-grow: 1;
}
</style>
