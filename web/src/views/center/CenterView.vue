<template>
  <div class="common-layout">
    <el-container class="container">
      <el-header class="header">
        <NavBar></NavBar>
      </el-header>
      <el-container>
        <el-aside class="aside">
          <AsideBar></AsideBar>
        </el-aside>
        <el-main class="main">
          <router-view class="center"></router-view>
        </el-main>
      </el-container>
    </el-container>
  </div>
</template>
  
<script lang="ts" setup>
import NavBar from '@/components/NavBar.vue'
import AsideBar from '@/components/AsideBar.vue'

</script>
  
<style lang="scss" scoped>
.container {
  height: 100vh; // 保证页面没有滚动条
}

.header {
  background: linear-gradient(180deg, #f5f4f6, #e6ebf7);
  height: 6vh;
  margin: 0;
  padding: 0;
}

.aside {
  background: radial-gradient(circle at 100% 0, #efe8eb 0, rgba(239, 232, 235, 0) 33%),
    radial-gradient(circle at 100% 25%, #e8ebea 0, hsla(160, 7%, 92%, 0) 39%),
    radial-gradient(circle at 100% 36%, hsla(160, 7%, 92%, .6) 0, hsla(160, 7%, 92%, 0) 38%),
    linear-gradient(180deg, #efeaef, #dcdcf5 99%);
  width: 200px;
  margin: 0;
  padding: 0;
}

.main {
  background: linear-gradient(180deg, #f5f4f6, #e6ebf7);
  margin: 0;
  padding: 0;
}
</style>
  