import { useState, useEffect } from "react"
import { askQuestion } from "../../services/api"


function getWelcomeContent(business, client) {

  const data = {
    hotel: {
      title: `Welcome to ${client} Hotel Assistant`,
      description: [
        "Ask about rooms, facilities, dining, and services.",
        "Get instant answers about your stay."
      ],
      suggestions: [
        "What are check-in timings?",
        "Do you offer parking?",
        "What facilities are available?",
        "Is breakfast included?"
      ]
    },

    restaurant: {
      title: `Welcome to ${client} Restaurant Assistant`,
      description: [
        "Explore menu, dishes, and services.",
        "Ask about timings and availability."
      ],
      suggestions: [
        "What are popular dishes?",
        "What are opening hours?",
        "Do you have vegetarian options?",
        "Do you serve late night food?"
      ]
    },

    clinic: {
      title: `Welcome to ${client} Clinic Assistant`,
      description: [
        "Ask about services, doctors, and timings.",
        "Get quick healthcare information."
      ],
      suggestions: [
        "What are consultation hours?",
        "Do you offer diagnostic services?",
        "Are appointments available?",
        "Do you provide health checkups?"
      ]
    }
  }

  return data[business] || data["hotel"]
}


export default function ChatWindow({business, client, sidebarOpen, setSidebarOpen}) {

  const [messages, setMessages] = useState([])
  const [loading, setLoading] = useState(false)
  const [suggestions, setSuggestions] = useState([])
  // const [business, setBusiness] = useState("hotel")
  const [uploading, setUploading] = useState(false)

  // const [client, setClient] = useState("")
  const [clients, setClients] = useState([])

  const isEmpty = messages.length === 0

  /* ------------------------------
     LOAD CLIENTS AUTOMATICALLY
  ------------------------------ */

  useEffect(()=>{

    fetch("http://localhost:8000/documents")
      .then(res=>res.json())
      .then(data=>{

        const businessClients = data[business] || []

        setClients(businessClients)
        setMessages([])
        setSuggestions([])
        setLoading(false)

        if(businessClients.length > 0){
          setClient(businessClients[0]) // auto select first
        } else {
          setClient("")
        }

      })
      .catch(()=>setClients([]))

  },[business, client])

  /* ------------------------------
     SEND MESSAGE
  ------------------------------ */

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

      setMessages([
        ...newMessages,
        {
          role: "assistant",
          content: data.answer || "I cannot find this information in the provided documents."
        }
      ])

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

      await fetch(`http://localhost:8000/${business}/${client}/upload`, {
        method: "POST",
        body: formData
      })

      alert(`Document uploaded for ${business}/${client}`)

    } catch (err) {

      alert("Upload failed")

    }

    setUploading(false)
  }

  return (

    

    <div className="chat-container">

      <div className="top-bar">
        <button onClick={()=>setSidebarOpen(prev => !prev)}>
          ☰
        </button>
        <h2>{business} Assistant</h2>
      </div>




      {isEmpty && (
  <div className="welcome-box">

    <h2>{getWelcomeContent(business, client).title}</h2>

    {getWelcomeContent(business, client).description.map((d,i)=>(
      <p key={i}>{d}</p>
    ))}

    <div className="welcome-suggestions">
      {getWelcomeContent(business, client).suggestions.map((s,i)=>(
        <button key={i} onClick={()=>sendMessage(s)}>
          {s}
        </button>
      ))}
    </div>

  </div>
)}

      
      {/* CHAT MESSAGES */}

      <div className="chat-window">

        {messages.map((m,i)=>(

          <div key={i} className={m.role}>

            {m.content.split("\n").map((line, idx)=>(

              <div key={idx}>{line}</div>

            ))}

          </div>

        ))}

      </div>

      {loading && <div className="thinking">Assistant is thinking...</div>}

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

      {/* INPUT */}

      <input
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

    </div>
  )
}