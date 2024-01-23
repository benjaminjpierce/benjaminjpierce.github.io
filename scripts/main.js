


const myImage = document.querySelector("img");

myImage.onclick = () => {
  const mySrc = myImage.getAttribute("src");
  if (mySrc === "images/me.jpeg") {
    myImage.setAttribute("src", "images/krentzman.jpeg");
  } else {
    myImage.setAttribute("src", "images/me.jpeg");
  }
};

let myButton = document.querySelector("button");
let myHeading = document.querySelector("h1");

function setUserName() {
    const myName = prompt("Please enter your name.");
    if (!myName) {
      setUserName();
    } else {
      localStorage.setItem("name", myName);
      myHeading.textContent = `Welcome, ${myName}`;
    }
  }
  
  
if (!localStorage.getItem("name")) {
  setUserName();
} else {
  const storedName = localStorage.getItem("name");
  myHeading.textContent = `Welcome back, ${storedName}`;
}

myButton.onclick = () => {
    setUserName();
  };
  

