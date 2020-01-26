const Z = 32;
const BATCH = 64;

let socketio = io.connect();

let generate_image = msg => {
  console.log(msg);

  let z = [];
  for (let i = 0; i < BATCH; i++) {
    let zz = [];
    for (let j = 0; j < Z; j++) {
      zz.push(Math.random());
    }
    z.push(zz);
  }

  const data = {
    z: z
  };

  console.log(data);

  const method = "POST";
  const body = JSON.stringify(data);
  const headers = {
    Accept: "application/json",
    "Content-Type": "application/json"
  };
  fetch("http://localhost:9824/api/generate", {
    method,
    headers,
    body,
    mode: "cors"
  })
    .then(res => res.json())
    .then(decode_images)
    .catch(console.error);
};

let decode_images = res => {
  let cnt = 0;
  const loop = setInterval(() => {
    show_images(res.base64[cnt]);
    cnt++;
    if (cnt == 64) {
      clearInterval(loop);
    }
  }, 50);
};

let show_images = b64 => {
  document.getElementById("image").src = `data:image/png;base64,${b64}`;
};

let oscSend = (address, msg) => {
  let obj = {};
  obj.address = address;
  obj.args = msg;
  io.emit("message", JSON.stringify(obj));
  console.log(`sent: on address: ${obj.address}`);
  console.dir(obj);
  return;
};

$("#osc_form").submit(() => {
  oscSend($("#osc_address").val(), $("#osc_msg").val());
  return false;
});

$("#message_form").submit(() => {
  socketio.send($("#input_msg").val());
  $("#input_msg").val("");
  return false;
});

socketio.on("message", msg => {
  console.log(`received msg: ${msg}`);
  $("#messages").prepend($("<li>").text(msg));
  generate_image(msg);
});
