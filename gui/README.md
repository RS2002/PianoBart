# PianoBart Web Demo

## Usage

### Backend

下载[fluidSynth](https://github.com/FluidSynth/fluidsynth)，将`fluidsynth.exe`所在的`bin`目录添加到环境变量中。

下载带有钢琴音色的SF2文件（例如[MuseScore的古典钢琴音色](https://freepats.zenvoid.org/Piano/YDP-GrandPiano/YDP-GrandPiano-SF2-20160804.tar.bz2)）放入项目路径 `gui\backend\default.sf2`。或在 `gui/backend/app.py`中修改sf2路径：

```python
sf2 = os.path.join(current_file_path, 'default.sf2')
```

启动后端服务器：

```shell
pip install -r gui/backend/requirements.txt
python -m gui.backend.app
```

### Frontend

#### 1、安装 Node.js 和 npm：

首先，确保你的系统上已经安装了 Node.js 和 npm。你可以从 [Node.js 官网](https://nodejs.org/) 下载并安装它们。安装完成后，你可以通过以下命令验证是否成功安装：

```shell
node -v
npm -v
```

#### 2. 安装 Vue CLI：

Vue CLI 是一个用于创建和管理 Vue.js 项目的官方命令行工具。安装 Vue CLI 可以让你更轻松地启动和管理 Vue.js 项目。运行以下命令来安装 Vue CLI：

```shell
npm install -g @vue/cli
```

安装完成后，你可以使用以下命令验证安装：

```shell
vue --version
```

#### 3、运行Vue

```shell
cd gui/frontend
npm install
npm run serve
```



