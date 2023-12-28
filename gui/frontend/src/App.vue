<template>
  <div>
    <router-view :key="$route.fullPath"></router-view>
  </div>
</template>

<script>
  export default {
    methods: {
      disableRTL() {
        if (!this.$rtl.isRTL) {
          this.$rtl.disableRTL();
        }
      },
      toggleNavOpen() {
        let root = document.getElementsByTagName('html')[0];
      }
    },
    mounted() {
      this.$watch('$route', this.disableRTL, { immediate: true });
    }
  };
</script>

<style lang="scss"></style>
