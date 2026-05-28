graph TD
    %% ================= NODES =================
    START((START))
    retrieve_ltm["retrieve_ltm"]
    retrieve_kg["retrieve_kg"]
    query_rewriter["query_rewriter"]
    tool_retriever["tool_retriever"]
    router["router"]
    mcp_executor["mcp_executor"]
    generator["generator"]
    summarizer["summarizer"]
    kg_manager["kg_manager"]
    END((END))

    %% ================= STORAGE =================
    LTM[("Long Term Memory<br/>(Qdrant + SQLite)")]
    KG[("Knowledge Graph<br/>(Temporal KG)")]
    ToolsDB[("Tool Registry<br/>(Qdrant)")]
    MCP[("MCP Servers<br/>(External)")]

    %% ================= WORKFLOW (Solid Lines) =================
    %% Indices: 0 to 13
    START --> retrieve_ltm
    START --> retrieve_kg
    retrieve_ltm --> query_rewriter
    retrieve_kg --> query_rewriter
    query_rewriter --> tool_retriever
    tool_retriever --> router
    router -->|"has tool_calls"| mcp_executor
    router -->|"no tool_calls & with tool result"| generator
    mcp_executor --> router
    generator --> summarizer
    generator --> kg_manager
    summarizer --> END
    kg_manager --> END
    generator --> END

    %% ================= DATA FLOW: WRITES (Red Dashed) =================
    %% Indices: 14 to 17
    MCP -. "Write (initialize & register tools)" .-> ToolsDB
    summarizer -. "Write (store conversation L1)" .-> LTM
    summarizer -. "Write (store facts L2)" .-> LTM
    kg_manager -. "Write (extract & update)" .-> KG

    %% ================= DATA FLOW: READS (Black Dashed) =================
    %% Indices: 18 to 22
    tool_retriever -. "Read (vector search Top-K)" .-> ToolsDB
    router -. "Read (vector search Top-K)" .-> ToolsDB
    retrieve_ltm -. "Read (semantic search)" .-> LTM
    retrieve_kg -. "Read (semantic search)" .-> KG
    mcp_executor -. "Call (execute tool)" .-> MCP

    %% ================= LINK STYLES =================
    %% Style for Write Flows (Red, Thick Dashed)
    linkStyle 14,15,16,17 stroke:red, stroke-width:3px, stroke-dasharray: 5 5;
    
    %% Style for Read/Call Flows (Black, Thick Dashed)
    linkStyle 18,19,20,21,22 stroke:black, stroke-width:3px, stroke-dasharray: 5 5;
