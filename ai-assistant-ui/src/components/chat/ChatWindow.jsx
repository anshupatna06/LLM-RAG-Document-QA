import { useState, useEffect, useRef } from "react"
import { askQuestion } from "../../services/api"
import VoiceButton from "../shared/VoiceButton"
import MessageList from "../shared/MessageList"
import WelcomeBox from "../shared/WelcomeBox"


// function getWelcomeContent(business, client) {

//   const data = {
//     hotel: {
//       title: `Welcome to ${client} AASHRAM AI Concierge`,
//       description: [
//         "Ask about rooms, dining & services.",
//         "Get instant answers."
//       ],
//       suggestions: [
//         "Check-in time?",
//         "Food menu?",
//         "Room availability?",
//         "WiFi & parking?"
//       ]
//     },

//     restaurant: {
//       title: `Welcome to ${client} Restaurant Assistant`,
//       description: [
//         "Explore menu, dishes, and services.",
//         "Ask about timings and availability."
//       ],
//       suggestions: [
//         "What are popular dishes?",
//         "What are opening hours?",
//         "Do you have vegetarian options?",
//         "Do you serve late night food?"
//       ]
//     },

//     clinic: {
//       title: `Welcome to ${client} Clinic Assistant`,
//       description: [
//         "Ask about services, doctors, and timings.",
//         "Get quick healthcare information."
//       ],
//       suggestions: [
//         "What are consultation hours?",
//         "Do you offer diagnostic services?",
//         "Are appointments available?",
//         "Do you provide health checkups?"
//       ]
//     }
//   }

//   return data[business] || data["hotel"]
// }




export default function ChatWindow({business, client, sidebarOpen, setSidebarOpen}) {

  const [messages, setMessages] = useState([])
  const [loading, setLoading] = useState(false)
  const [suggestions, setSuggestions] = useState([])
  // const [business, setBusiness] = useState("hotel")
  const [uploading, setUploading] = useState(false)

  // const [client, setClient] = useState("")
  const [clients, setClients] = useState([])

  const isEmpty = messages.length === 0
  const chatEndRef = useRef(null)

  const [listening, setListening] = useState(false)

  const [currentContact, setCurrentContact] = useState(null)
  const [roomNumber, setRoomNumber] = useState("")

  /* ------------------------------
     LOAD CLIENTS AUTOMATICALLY
  ------------------------------ */

  // useEffect(()=>{

  //   fetch("http://localhost:8000")
  //     .then(res=>res.json())
  //     .then(data=>{

  //       const businessClients = data[business] || []

  //       setClients(businessClients)
  //       setMessages([])
  //       setSuggestions([])
  //       setLoading(false)

  //       if(businessClients.length > 0){
  //         setClient(businessClients[0]) // auto select first
  //       } else {
  //         setClient("")
  //       }

  //     })
  //     .catch(()=>setClients([]))

  // },[business, client])

  useEffect(() => {

    fetch("https://llm-rag-document-qa-3.onrender.com")
      .then(res=>res.json())
      .then(data=>{

        const businessClients = data[business] || []

        setClients(businessClients)
        setMessages([])
        setSuggestions([])
        setLoading(false)

        // 🔥 ONLY auto-select if client NOT already set
        if(!client && businessClients.length > 0){
          setClient(businessClients[0])
        }

      })
      .catch(()=>setClients([]))

  }, [business])   // 🔥 REMOVE client from dependency


  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages, loading])



  const startListening = () => {

    alert("Please allow microphone access for voice feature")

    const SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition

    if (!SpeechRecognition) {
      alert("Speech recognition not supported in this browser")
      return
    }

    const recognition = new SpeechRecognition()

    recognition.lang = "hi-IN" // 🔥 supports Hindi + Hinglish
    recognition.interimResults = false
    recognition.continous = false
    recognition.maxAlternatives = 1


    // setListening(true)

    // recognition.start()

    recognition.onstart = () => {
      console.log("🎤 Listening started")
    }

    recognition.onspeechstart = () => {
      console.log("🗣️ User started speaking")
    }

    recognition.onresult = (event) => {
      const text = event.results[0][0].transcript
      console.log("🎤 Heard:", text)

      setListening(false)

      sendMessage(text) // 🔥 directly send to your pipeline
      console.log("VOICE TEXT:", transcript)
    }

    recognition.onerror = (event) => {
      console.error("🎤 Speech Error:", event.error)
      setListening(false)
    
      if (event.error === "not-allowed") {
        alert("Please allow microphone access")
      } else if (event.error === "no-speech") {
        alert("No speech detected, try again")
      } else {
        alert("Voice error: " + event.error)
      }
    }

    recognition.onend = () => {
      setListening(false)
    }

    setListening(true)
    recognition.start()
  }


  /* ------------------------------
     SEND MESSAGE
  ------------------------------ */

  const speakText = (text) => {

    if (! text) return

    const speech = new SpeechSynthesisUtterance(text)

    // 🔥 Detect Hindi
    if (/[\u0900-\u097F]/.test(text)) {
      speech.lang = "hi-IN"
    } else {
    speech.lang = "en-IN"
    }

    window.speechSynthesis.speak(speech)
  }

  const sendMessage = async (text) => {

    if(!text || !client) return

    const newMessages = [
      ...messages,
      { role: "user", content: text }
    ]

    setMessages(newMessages)
    setLoading(true)
    setSuggestions([])

    try {

      const data = await askQuestion(text, business, client)

      setCurrentContact(data.contact)

      setMessages([
        ...newMessages,
        {
          role: "assistant",
          content: data.answer || "I cannot find this information in the provided documents.",
          actions:data.actions || []
        }
      ])

      speakText(data.answer)

      

      setSuggestions(data.suggestions || [])

    } catch (error) {

      setMessages([
        ...newMessages,
        {
          role: "assistant",
          content: "Something went wrong while processing your request."
        }
      ])
    }

    setLoading(false)
  }

  /* ------------------------------
     BUSINESS CHANGE
  ------------------------------ */

  const handleBusinessChange = (e) => {

    const newBusiness = e.target.value

    setBusiness(newBusiness)
    setMessages([])
    setSuggestions([])
  }

  /* ------------------------------
     DOCUMENT UPLOAD (FIXED)
  ------------------------------ */

  const uploadDocument = async (file) => {

    if(!file || !client) return

    const formData = new FormData()
    formData.append("file", file)

    setUploading(true)

    try {

      await fetch(`https://llm-rag-document-qa-3.onrender.com/${business}/${client}/upload`, {
        method: "POST",
        body: formData
      })

      alert(`Document uploaded for ${business}/${client}`)

    } catch (err) {

      alert("Upload failed")

    }

    setUploading(false)
  }


  // const handleAction = (action) => {

  //   if(action.type === "call"){
  //     window.location.href = "tel:+917488713865"   // hotel number
  //   }

  //   if(action.type === "assist_wifi"){
  //     sendMessage("I need help connecting to wifi")
  //   }

  //   if(action.type === "request_towel"){
  //     sendMessage("Please send a towel to my room")
  //   }

  //   if(action.type === "menu"){
  //     sendMessage("Show me the food menu")
  //   }

  //   if(action.type === "order_food"){
  //     sendMessage("I want to order food")
  //   }
  // }

  const handleAction = (action) => {

    const phone = currentContact?.phone
    const isMobile = /iPhone|Android/i.test(navigator.userAgent)
    const whatsapp = currentContact?.whatsapp

    // 📞 CALL
    if (action.type === "call") {
      if(isMobile){
        window.location.href = `tel:${phone}`
      }else{
        alert("please call using your phone.")
      }
    }

    // 💬 WHATSAPP
    if (action.type === "whatsapp") {

      fetch("https://llm-rag-document-qa-3.onrender.com/service-request", {

        method: "POST",

        headers: {
          "Content-Type": "application/json"
        },

        body: JSON.stringify({

          room: roomNumber || "unknown",
          request: action.request_type || "general",
          client_id: client
        })
      })
      .then(res => res.json())
      .then(data => {

        console.log("REQUEST STORED:", data)

        setMessages(prev => [

          ...prev,

          {
            role: "assistant",
            content:
              `✅ Request submitted successfully (${data.request.request_id})`
          }

        ])
      })
      .catch(err => console.log("REQUEST ERROR:", err))

      if (!currentContact?.whatsapp){
        alert("whatsapp contact not available")
        return
      }

      const cleanNumber = currentContact?.whatsapp?.replace("+", "")
      let safeMessage = action.message || "Hello, I need help."
      if (roomNumber.trim()){
        safeMessage = `Room ${roomNumber}: ${safeMessage}`
      }

      const message = encodeURIComponent(safeMessage)
      

      const url = `https://wa.me/${cleanNumber}?text=${message}`

      console.log("WHATSAPP URL:", url)
      console.log("ACTION:", action)
      console.log("CONTACT:", currentContact)

      window.open(url, "_blank")
    }

    // 🤖 INTERNAL ACTION
    if (action.type === "assist_wifi") {
      // alert(
      //   "Please open your device WiFi settings and connect to the hotel WiFi network. If you still need help, contact reception."
      // )
      // return

      setMessages(prev => [
        ...prev,
        {
            role: "assistant",
            content:
              "Please open your device WiFi settings and connect to the hotel WiFi network. If you still face issues, contact reception."
        }
    ])

    return
    }
  }

  console.log("CONTACT:", currentContact)

  return (

    

    <div className="chat-container">

      

      <div className="top-bar">
        <button onClick={()=>setSidebarOpen(prev => !prev)}>
          ☰
        </button>
        <h2>{business} Assistant</h2>
      </div>



      {isEmpty && (

        <WelcomeBox
          business={business}
          client={client}
          sendMessage={sendMessage}
        />

      )}


      
      {/* CHAT MESSAGES */}

      <MessageList
        messages={messages}
        handleAction={handleAction}
        chatEndRef={chatEndRef}
      />

      

      {loading && <div className="thinking">🤖 Assistant is thinking...</div>}

      {/* SUGGESTIONS */}

      {suggestions.length > 0 && (

        <div className="suggestions">

          {suggestions.map((s,i)=>(

            <button key={i} onClick={()=>sendMessage(s)}>
              {s}
            </button>

          ))}

        </div>

      )}

      
    
      <div className = "input-container">

        <input
          type="text"
          placeholder="Room Number (optional)"
          value={roomNumber}
          onChange={(e) => setRoomNumber(e.target.value)}
          className="room-input"
        />


        {/* INPUT */}
        <input
          className="message-input"
          placeholder={`Ask the ${business}/${client} assistant...`}
          onKeyDown={(e)=>{

            if(e.key==="Enter"){

              const text = e.target.value.trim()

              if(!text) return

              sendMessage(text)

              e.target.value=""
            }

          }}
        />

        {/* 🎤 MIC BUTTON */}
        <VoiceButton
          listening={listening}
          startListening={startListening}
        />

      </div>
    </div>
  )
}