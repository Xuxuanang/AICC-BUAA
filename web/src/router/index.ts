import { createRouter, createWebHistory } from 'vue-router'
import UserAccountViewVue from '@/views/user/account/UserAccountView.vue'
import UserAccountLoginViewVue from '@/views/user/account/UserAccountLoginView.vue'
import UserAccountRegisterViewVue from '@/views/user/account/UserAccountRegisterView.vue'
import CenterViewVue from '@/views/center/CenterView.vue'
import CodeTransformVue from '@/views/center/CodeTransform.vue'
import TransformDetailViewVue from '@/views/center/TransformDetailView.vue'
import store from '../store/user'

const routes = [
  {
    path: "/",
    name: "home",
    redirect: "/center/transform/",
    meta: {
      requestAuth: true,
    }
  },
  {
    path: "/user/account/",
    name: "user_account",
    component: UserAccountViewVue,
    meta: {
      requestAuth: false,
    },
    children: [
      {
        path: "login",
        name: "login",
        component: UserAccountLoginViewVue,
        meta: {
          requestAuth: false,
        }
      },
      {
        path: "register",
        name: "register",
        component: UserAccountRegisterViewVue,
        meta: {
          requestAuth: false,
        }
      }
    ]
  },
  {
    path: "/center/",
    name: "center",
    component: CenterViewVue,
    meta: {
      requestAuth: true,
    },
    children: [
      {
        path: "transform",
        name: "transform",
        component: CodeTransformVue,
        meta: {
          requestAuth: true,
        }
      },
      {
        path: "detail",
        name: "detail",
        component: TransformDetailViewVue,
        meta: {
          requestAuth: true,
        }
      }
    ]
  },
  {
    path: "/:catchAll(.*)",
    redirect: "/404/"
  }
]


const router = createRouter({
  history: createWebHistory(process.env.BASE_URL),
  routes
})

router.beforeEach((to, from, next) => {
  if (to.meta.requestAuth && !store.state.is_login) {
    next({name: "login"});
  } else {
    next();
  }

})

export default router
