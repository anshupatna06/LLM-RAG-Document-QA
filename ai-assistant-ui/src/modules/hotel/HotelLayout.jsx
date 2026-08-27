
// export default function HotelLayout(props) {

//   const [hotelPanelOpen, setHotelPanelOpen] = useState(false)
//   return (

//     <div className="hotel-main-layout">
      
//       {sidebarOpen && (
//           <div
//             className="sidebar-overlay"
//             onClick={() => setHotelPanelOpen(false)}
//           />
//         )}

//       {/* MAIN CHAT */}

//       <HotelChatWindow
//         {...props}
//         sidebarOpen={sidebarOpen}
//         setSidebarOpen={setSidebarOpen}
//       />

      

//       {/* HOTEL SIDE PANELS */}

//       <div
//         className={`hotel-sidepanels ${
//           sidebarOpen ? "open" : ""
//         }`}
//       >

//         <button
//           className="close-sidebar"
//           onClick={() => setSidebarOpen(false)}
//         >
//           ✕
//         </button>

//         <ExplorePanel client={props.client} />

//         <BranchesPanel client={props.client} />
//       </div>

//     </div>
//   )
// }

// export default function HotelLayout({
//   business,
//   client,
//   adminSidebarOpen,
//   setAdminSidebarOpen,
//   explorePanelOpen,
//   setExplorePanelOpen
// }) {

//   const [hotelPanelOpen, setHotelPanelOpen] = useState(false);

//   const openExplorePanel = () => {
//     setExplorePanelOpen(true)
//   }

//   return (
//     <div className="hotel-main-layout">

//       {hotelPanelOpen && (
//         <div
//           className="sidebar-overlay"
//           onClick={() => setHotelPanelOpen(false)}
//         />
//       )}

//       <HotelChatWindow
//         business={business}
//         client={client}

//         adminSidebarOpen={adminSidebarOpen}
//         setAdminSidebarOpen={setAdminSidebarOpen}

//         explorePanelOpen={explorePanelOpen}
//         setExplorePanelOpen={setExplorePanelOpen}

//         onOpenExplore={openExplorePanel}

//         // hotelPanelOpen={hotelPanelOpen}
//         // setHotelPanelOpen={setHotelPanelOpen}
//       />

//       <div
//         className={`hotel-sidepanels ${
//           hotelPanelOpen ? "open" : ""
//         }`}
//       >
//         <aside className="hotel-sidepanels right-explore-panel open">

//           <div className="right-panel-header">

//             <button
//               className="close-sidebar"
//               onClick={() => setHotelPanelOpen(false)}
//             >
//               ✕
//             </button>
//           </div>  

//           <ExplorePanel client={client} />

//           <BranchesPanel client={client} />
//         </aside>

//       </div>

//     </div>
//   );
// }

import HotelChatWindow from "./HotelChatWindow"

import ExplorePanel
from "../../components/hotel/ExplorePanel"

import BranchesPanel
from "../../components/hotel/BranchesPanel"


export default function HotelLayout({
  business,
  client,
  adminSidebarOpen,
  setAdminSidebarOpen,
  explorePanelOpen,
  setExplorePanelOpen
}) {

  const openExplorePanel = () => {
    setExplorePanelOpen(true)
  }

  console.log(
    "explorePanelOpen:",
    explorePanelOpen
  )

  return (

    <div className="hotel-main-layout">

      {/* MAIN CHAT */}

      <HotelChatWindow
        business={business}
        client={client}

        adminSidebarOpen={adminSidebarOpen}
        setAdminSidebarOpen={setAdminSidebarOpen}

        explorePanelOpen={explorePanelOpen}
        setExplorePanelOpen={setExplorePanelOpen}

        onOpenExplore={openExplorePanel}
      />


      {/* RIGHT EXPLORE PANEL */}

      <aside
        className={`right-explore-panel ${
          explorePanelOpen ? "open" : ""
        }`}
      >

        <div className="right-panel-header">

          <button
            className="close-sidebar"
            onClick={() => {
              console.log("CLOSING EXPLORE PANEL")
              setExplorePanelOpen(false)
            }}
          >
            ✕
          </button>

        </div>

        <ExplorePanel client={client} />

        <BranchesPanel client={client} />

      </aside>

    </div>
  )
}