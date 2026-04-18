// import ChatWindow from "./components/chat/ChatWindow"
// import Sidebar from "./components/Sidebars/Sidebar"

// function App() {

//   return (

//     <div className="app-layout">
      
//       <h1>Hotel AI Assistant</h1>

//       <Sidebar/>

//       <ChatWindow/>

//     </div>

//   )
// }

// export default App

import ChatWindow from "./components/chat/ChatWindow"
import Sidebar from "./components/Sidebars/Sidebar"
import { useState } from "react"

function App() {

  const [business, setBusiness] = useState("hotel")
  const [client, setClient] = useState("taj")
  const [sidebarOpen, setSidebarOpen] = useState(true)

  return (
    <div className="app-layout">

      {/* SIDEBAR */}
      <Sidebar
        business={business}
        setBusiness={setBusiness}
        client={client}
        setClient={setClient}
        sidebarOpen={sidebarOpen}
      />

      {/* CHAT */}
      <ChatWindow
        business={business}
        client={client}
        sidebarOpen={sidebarOpen}
        setSidebarOpen={setSidebarOpen}
      />

    </div>
  )
}

export default App