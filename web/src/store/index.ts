import { createStore } from 'vuex'
import ModuleUser from './user'

export default createStore({
  state: {
    page_id: 0,
  },
  mutations: {
    updatePageId(state, page_id) {
      state.page_id = page_id;
    }
  },
  actions: {
    updatePageId(context, data) {
      context.commit("updatePageId", data.page_id);
    }
  },
  modules: {
    user: ModuleUser
  }
})
