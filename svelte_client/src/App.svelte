<style>
  #img_capt_d, #img_upl_d {
    display: none;
  }

  #video {
      transform: rotateY(180deg);
      -ms-transform:rotateY(180deg); /* IE 9 */
      -webkit-transform:rotateY(180deg); /* Safari and Chrome */
      -moz-transform:rotateY(180deg); /* Firefox */
      -o-transform:rotateY(180deg); /* Opera */
  }

</style>

<script>

  let videoSource = null;
  let any_btn_clicked = false;
  let video = null;
  // let img_inp_canvas = document.getElementById("img_inp_canvas");
  let emotion = "";

  // let height = 0;
  // let width = 0;

  async function getEmotion(data, mirrored=false) {

    fetch('./api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({data})
      }).then(response => response.json()).then(data => {

        console.log(data);

        if (data.segmentation_bounds) {

            let bounds = data.segmentation_bounds;

            let img_inp_canvas = document.getElementById("img_inp_canvas");
            const context = img_inp_canvas.getContext("2d");

            if (mirrored) {
                context.translate(img_inp_canvas.width, 0);
                context.scale(-1,1);
            }

            context.beginPath();
            context.rect(bounds.x, bounds.y, bounds.w, bounds.h);
            context.lineWidth = 4;
            context.strokeStyle = 'red';
            context.stroke();
        }

        emotion = data.emotion;

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

    let img_inp_canvas = document.getElementById("img_inp_canvas");
    if (img_inp_canvas) {
      img_inp_canvas.setAttribute('height', 0);
      img_inp_canvas.setAttribute('width', 0);
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
    // height = video.videoHeight;
    // width = video.videoWidth;

    let cam_d = document.getElementById("camera_d");
    cam_d.style.display = "none";

    // img_inp_canvas = document.getElementById("img_inp_canvas");
    // img_inp_canvas.setAttribute('height', height);
    // img_inp_canvas.setAttribute('width', width);
    //
    // const context = img_inp_canvas.getContext("2d");
    //
    // context.translate(img_inp_canvas.width, 0);
    // context.scale(-1,1);
    //
    // context.drawImage(video, 0, 0, width, height);

    const data = displayImage(video.videoHeight, video.videoWidth, video, true);

    // let img_inp_canvas = document.getElementById("img_inp_canvas");
    // const data = img_inp_canvas.toDataURL("image/png");

    // Stop video after image capture
    if (video) {
      const tracks = video.srcObject.getTracks();
      tracks.forEach(track => track.stop());
      video.pause();
    }

    getEmotion(data, true);

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

    let img_file = e.target.files[0];

    let reader = new FileReader();
    reader.readAsDataURL(img_file);

    reader.onloadend = e => {
        let image = new Image();
        image.src = e.target.result;

        console.log("Image height: " + image.height);
        console.log("Image width: " + image.width);

        image.onload = ev => {
            const data = displayImage(image.height, image.width, image);
            getEmotion(data);
        }

    };
  }

  function displayImage(height, width, data, mirror=false) {

      let img_inp_canvas = document.getElementById("img_inp_canvas");

      console.log("displayImage height: " + height);
      console.log("displayImage width: " + width);

      img_inp_canvas.setAttribute('height', height);
      img_inp_canvas.setAttribute('width', width);

      const context = img_inp_canvas.getContext("2d");

      if (mirror) {
          context.translate(img_inp_canvas.width, 0);
          context.scale(-1,1);
      }

      context.drawImage(data, 0, 0, width, height);

      return img_inp_canvas.toDataURL("image/png");
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

        <div class="col-md-12 col-lg-8 p-2 my-4">

            <div class="row text-center pt-2">
                <div class="btn-group" role="group">
                    <button id="img_capt_b" on:click={enableVideoCamera} disabled={any_btn_clicked} class="btn btn-sm btn-primary m-1">Capture Image</button>
                    <button id="img_upl_b" on:click={enableFileUpload} disabled={any_btn_clicked} class="btn btn-sm btn-primary m-1">Upload Image</button>
                    <button id="cancel_b" on:click={resetState} disabled={!any_btn_clicked} class="btn btn-sm btn-danger m-1">Reset</button>
                </div>
            </div>

            <div id="inp_d" class="row">
                <div id="img_capt_d" class="container-fluid">
                    <div id="camera_d" class="m-5 align-items-center text-center">
                      <!-- svelte-ignore a11y-media-has-caption -->
                      <video id="video" class="" bind:this={videoSource} />
                      <button id="img_snap_b" on:click={takeSnapshot} class="btn btn-sm btn-secondary">Take Photo</button>
                    </div>

<!--                    <div id="photo_d" class="m-5 align-items-center text-center">-->
<!--                        &lt;!&ndash;  This is where the image will be displayed after capture&ndash;&gt;-->
<!--                        <canvas id="img_inp_canvas"></canvas>-->
<!--                    </div>-->
                </div>

                <div id="img_upl_d" class="container-fluid align-items-center text-center">
                    <input id="upl_img" type="file" accept="image/*" on:change={(e)=>uploadFile(e)} class="form-control my-3 bg-dark text-light"/>
                </div>

                <canvas id="img_inp_canvas"></canvas>

            </div>

        </div>

        <div class="col-md-12 col-lg-4 p-2">
            <div id="result_d" class="align-items-center text-center">
                <h2 class="fs-4 text">Emotion Detected</h2>
                <p id="result_t" class="fs-5 text">{emotion}</p>
            </div>
        </div>
    </div>
