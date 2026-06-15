import HotelChatWindow from "./HotelChatWindow"

import ExplorePanel
from "../../components/hotel/ExplorePanel"

import BranchesPanel
from "../../components/hotel/BranchesPanel"

import { useState } from "react" 
export default function HotelLayout(props) {

  const [sidebarOpen, setSidebarOpen] = useState(false)
  return (

    <div className="hotel-main-layout">
      
      {sidebarOpen && (
          <div
            className="sidebar-overlay"
            onClick={() => setSidebarOpen(false)}
          />
        )}

      {/* MAIN CHAT */}

      <HotelChatWindow
        {...props}
        sidebarOpen={sidebarOpen}
        setSidebarOpen={setSidebarOpen}
      />

      

      {/* HOTEL SIDE PANELS */}

      <div
        className={`hotel-sidepanels ${
          sidebarOpen ? "open" : ""
        }`}
      >

        <ExplorePanel client={props.client} />

        <BranchesPanel client={props.client} />
      </div>

    </div>
  )
}