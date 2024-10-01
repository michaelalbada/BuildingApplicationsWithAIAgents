from autogen.agentchat.contrib.web_surfer import WebSurferAgent

web_surfer = WebSurferAgent(
    "web_surfer",
    llm_config=llm_config,
    summarizer_llm_config=summarizer_llm_config,
    browser_config={"viewport_size": 4096},
)

user_proxy = autogen.UserProxyAgent(
    "user_proxy",
    human_input_mode="NEVER",
    code_execution_config=False,
    default_auto_reply="",
    is_termination_msg=lambda x: True,
)