<style>
  #img_capt_d, #img_upl_d {
    display: none;
  }

</style>

<script>
  let videoSource = null;
  let any_btn_clicked = false;
  let video = null;
  let canvas = null;
  let emotion = "";

  let width = 0;
  let height = 0;

  async function getEmotion(data) {

    fetch('./api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({data})
      }).then(response => response.text()).then(data => {
        emotion = data;

      }).catch((error) => {
        console.error('Error:', error);
      });
  }

  function resetState() {

    // Button reset
    any_btn_clicked = false;

    // Video Camera reset
    let img_capt_d = document.getElementById("img_capt_d");
    img_capt_d.style.display = "none";

    if (video) {
      const tracks = video.srcObject.getTracks();
      tracks.forEach(track => track.stop());
      video.pause();
    }

    let cam_d = document.getElementById("camera_d");
    cam_d.style.display = "block";

    if (canvas) {
      canvas.setAttribute('height', 0);
      canvas.setAttribute('width', 0);
    }

    // File Upload reset
    let img_upl_d = document.getElementById("img_upl_d");
    img_upl_d.style.display = "none";

    // Result reset
      emotion = "";

  }
  const enableVideoCamera = async () => {
    try {
      video = document.getElementById("video");
      canvas = document.getElementById("img_canvas");

      any_btn_clicked = true;

      let img_capt_d = document.getElementById("img_capt_d");
      let img_upl_d = document.getElementById("img_upl_d");

      img_capt_d.style.display = "flex";
      img_upl_d.style.display = "none";

      videoSource.srcObject = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false
      });
      videoSource.play();

    } catch (error) {
      console.log(error);
      resetState();
    }
  };

  function takeSnapshot() {

    video = document.getElementById("video");
    height = video.videoHeight;
    width = video.videoWidth;

    let cam_d = document.getElementById("camera_d");
    cam_d.style.display = "none";

    canvas.setAttribute('height', height);
    canvas.setAttribute('width', width);

    const context = canvas.getContext("2d");

    context.drawImage(video, 0, 0, width, height);

    const data = canvas.toDataURL("image/png");

    getEmotion(data);

  }

  function enableFileUpload() {
    any_btn_clicked = true;
    console.log("enableFileUpload");

    let img_capt_d = document.getElementById("img_capt_d");
    let img_upl_d = document.getElementById("img_upl_d");

    img_upl_d.style.display = "flex";
    img_capt_d.style.display = "none";

  }

  const uploadFile =(e) => {

    let img = e.target.files[0];

    let reader = new FileReader();
    reader.readAsDataURL(img);

    reader.onload = e => {
         getEmotion(e.target.result);
    };
  }

</script>

    <div class="row align-items-start">
        <h1 class="display-6 text-center m-3">Facial Emotion Recognition</h1>
        <div class="container m-2">
            <p class="m-0 fs-5">Usage:</p>
            <ul class="list-group">
                <li class="list-group-item-text"><small>Click <span class="badge bg-primary">Capture Image</span> to enable the camera and capture an image.</small></li>
                <li class="list-group-item-text"><small>Or click <span class="badge bg-primary">Upload Image</span> to upload an image from your device.</small></li>
                <li class="list-group-item-text"><small>The image will be processed and the emotion will be displayed.</small></li>
                <li class="list-group-item-text"><small>Click <span class="badge bg-danger">Reset</span> to start over.</small></li>
            </ul>
        </div>

        <div class="col col-sm-12 col-md-8 p-2 my-4">
            <div class="row text-center align-items-start pt-2">
              <button id="img_capt_b" on:click={enableVideoCamera} disabled={any_btn_clicked} class="col btn btn-sm btn-primary mx-3">Capture Image</button>
              <button id="img_upl_b" on:click={enableFileUpload} disabled={any_btn_clicked} class="col btn btn-sm btn-primary mx-3">Upload Image</button>
              <button id="cancel_b" on:click={resetState} disabled={!any_btn_clicked} class="col btn btn-sm btn-danger mx-3">Reset</button>
            </div>

            <div id="inp_d" class="row">
                <div id="img_capt_d" class="container-fluid">
                    <div id="camera_d" class="m-5 align-items-center text-center">
                      <!-- svelte-ignore a11y-media-has-caption -->
                      <video id="video" class="object-fit-contain" bind:this={videoSource} />
                      <button id="img_snap_b" on:click={takeSnapshot} class="btn btn-sm btn-secondary">Take Photo</button>
                    </div>

                    <div id="photo_d" class="m-5 align-items-center text-center">
                        <!--  This is where the image will be displayed after capture-->
                        <canvas id="img_canvas"></canvas>
                    </div>
                </div>

                <div id="img_upl_d" class="container-fluid align-items-center text-center">
                    <input id="upl_img" type="file" accept="image/*" on:change={(e)=>uploadFile(e)} class="form-control my-3 bg-dark text-light"/>
                </div>
            </div>
        </div>

        <div class="col col-sm-12 col-md-4 p-2">
            <div id="result_d" class="align-items-center text-center">
                <h2 class="fs-4 text">Emotion Detected</h2>
                <p id="result_t" class="fs-5 text">{emotion}</p>
            </div>
        </div>
    </div>
