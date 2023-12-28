<template>
    <!-- <side-bar>
      <template slot="links">
        <sidebar-link to="/dashboard" :name="$t('sidebar.dashboard')" icon="tim-icons icon-chart-pie-36"/>
        <sidebar-link to="/icons" :name="$t('sidebar.overview')" icon="tim-icons icon-atom"/>
        <sidebar-link to="/maps" :name="$t('sidebar.maps')" icon="tim-icons icon-pin"/>
        <sidebar-link to="/notifications" :name="$t('sidebar.notifications')" icon="tim-icons icon-bell-55"/>
        <sidebar-link to="/generate" :name="$t('sidebar.diagnose')" icon="tim-icons icon-puzzle-10"/>
        <sidebar-link to="/table-list" :name="$t('sidebar.tableList')" icon="tim-icons icon-puzzle-10"/>
        <sidebar-link to="/typography" :name="$t('sidebar.typography')" icon="tim-icons icon-align-center"/>
        <sidebar-link to="/dashboard?enableRTL=true" :name="$t('sidebar.rtlSupport')" icon="tim-icons icon-world"/>
      </template>
    </side-bar> -->
    <div class="main-panel">
      <!-- <top-navbar></top-navbar> -->

      <dashboard-content>

      </dashboard-content>

      <content-footer></content-footer>
    </div>
</template>
<style lang="scss">
</style>
<script>
import ContentFooter from "./ContentFooter.vue";
import DashboardContent from "./Content.vue";
export default {
  components: {
    ContentFooter,
    DashboardContent,
  },
  methods: {
    // toggleSidebar() {
    //   if (this.$sidebar.showSidebar) {
    //     this.$sidebar.displaySidebar(false);
    //   }
    // }
  }
};
</script>
