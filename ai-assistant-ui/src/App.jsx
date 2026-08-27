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

// import ChatWindow from "./components/chat/ChatWindow"
import RequestsPanel from "./components/hotel/RequestsPanel"
import HotelLayout from "./modules/hotel/HotelLayout"
// import ExplorePanel from "./components/hotel/ExplorePanel"
// import BranchesPanel from "./components/hotel/BranchesPanel"
import Sidebar from "./components/Sidebars/Sidebar"
import { useState } from "react"
import { useParams } from "react-router-dom"
import { useLocation } from "react-router-dom"
import { useEffect } from "react"

function App() {

  const params = useParams()
  const location = useLocation()

  const [business, setBusiness] = useState("hotel")
  const [client, setClient] = useState("taj")
  const [adminSidebarOpen, setAdminSidebarOpen] = useState(true)
  const [explorePanelOpen, setExplorePanelOpen] = useState(true)

  useEffect(() => {
    if (params.business && params.client) {
      setBusiness(params.business)
      setClient(params.client)
    }
  },  [params.business, params.client])

  return (

  <>
  
    {location.pathname === "/admin" ? (

      <RequestsPanel/>

    ) : (

      <div className="app-layout">

        {/* SIDEBAR */}
        {!(params.business && params.client) && (
          <Sidebar
            business={business}
            setBusiness={setBusiness}
            client={client}
            setClient={setClient}
            sidebarOpen={adminSidebarOpen}
          />
        )}

        {/* BUSINESS LAYOUT */}
        {business === "hotel" && (
          <HotelLayout
            business={business}
            client={client}
            adminSidebarOpen={adminSidebarOpen}
            setAdminSidebarOpen={setAdminSidebarOpen}

            explorePanelOpen={explorePanelOpen}
            setExplorePanelOpen={setExplorePanelOpen}
          />
        )}

      </div>

    )}

  </>
  )
}
export default App