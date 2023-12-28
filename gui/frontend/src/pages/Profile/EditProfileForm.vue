<!-- eslint-disable vue/no-mutating-props -->
<template>
  <card :style="{ textAlign: 'center' }">
    <h3 slot="header" class="title"><img width="48" height="48" src="https://img.icons8.com/color/48/piano.png" alt="piano"/> PianoBart - Your Piano Generator

    </h3>
    <div class="upload-box">
      <!-- 隐藏的文件输入框 -->
      <input type="file" ref="fileInput" style="display:none" @change="uploadMidi" accept=".midi, .midi, audio/midi">

      <!-- 上传按钮 -->
      <base-button 
        :style="{ marginTop: '-2%', fontSize: '1.7em', lineHeight: '1.5em' }"
        slot="footer"
        type="primary"
        fill
        @click="triggerFileInput">
          Choose Intro <i class="far fa-file-audio"></i>
      </base-button>



  </div>
    <div style="text-align: center;">

      <div class="intro" style="margin-bottom: 20px;">
          <av-waveform :style="{ marginTop: '2%' }" :audio-src="`http://localhost:5000/api/upload/${midiFile}.wav`" controls v-if="midiFile" ></av-waveform>
          <h4 :style="{ marginTop: '2%' }">{{ upload_message }}</h4>
      </div>

      <div style="margin-bottom: 20px;">
          <base-button :style="{ fontSize: '1.3em', lineHeight: '1.2em' }" slot="footer" type="primary" fill @click="generate()">Generate</base-button>
      </div>

      <div class="generated" style="margin-bottom: 20px;">
          <h4 style="margin-bottom: 10px; font-size: 1.5em; font-weight: bold; color: #4A4A4A;" controls v-if="output">Intro:</h4>
          <av-waveform  :audio-src="`http://localhost:5000/api/upload/${output}_intro.wav`" controls v-if="output" ></av-waveform>
          <h4 style="margin-bottom: 10px; font-size: 1.5em; font-weight: bold; color: #4A4A4A;" controls v-if="output">Generated:</h4>
          <av-waveform  :audio-src="`http://localhost:5000/api/output/${output}.wav`" controls v-if="output" ></av-waveform>
          <br>
          <h4 :style="{ marginTop: '2%' }">{{ generate_message }}</h4>
      </div>


      </div>
  </card>
</template>

<script>
import axios from "axios";
export default {
  data() {
    return {
      midiFile: null,
      model: 'pianobart',
      upload_message: null,
      generate_message:null,
      output: null
    };
  },
  methods: {
    // 触发文件输入框的方法
    triggerFileInput() {
      this.$refs.fileInput.click();
    },
    uploadMidi(event) {
      const formData = new FormData();
      formData.append('file', event.target.files[0]);
      this.output = null;
      this.midiFile = null;
      this.upload_message = null;
      this.generate_message = null;
      axios.post('http://localhost:5000/api/upload', formData)
        .then(response => {
          let midiUrl = 'http://localhost:5000/api/upload/' + response.data.filename;
          this.midiFile = response.data.filename;
          // 更新音频播放器的源
          // 这里可以添加处理 MIDI 文件的逻辑
          console.log('MIDI file uploaded:', midiUrl);
          this.upload_message = `Uploaded: ${this.midiFile}.mid`;
          // 可以在这里更新 UI，例如显示 MIDI 文件名或其他信息
        })
        .catch(error => console.error(error));
    },

    generate() {
      this.output = null;
      this.generate_message = null;
      if (this.midiFile && this.model) {
        this.generate_message = 'PianoBart is generating...';
        axios.get('http://localhost:5000/api/generate/' + this.model + '/' + this.midiFile)
        .then(response => {
          // 这里处理服务器的响应
          this.generate_message = 'Successfully generated!';
          console.log(this.generate_message);
          this.output = response.data.result;
          console.log(response.data.result);
        })
        .catch(error => {
          console.error(error);
          this.generate_message = null;
        });
      }
      else {
        this.generate_message = 'Upload a midi file / choose a model first.';
      }
  },
},
};
</script>
<style>


.upload-box {
  display: flex;
  justify-content: center;
  align-items: center;
  cursor: pointer;
  margin-left: auto; /* 尝试添加这行 */
  margin-right: auto; /* 尝试添加这行 */
}
</style>
