# from agent.utils.action_generator import generate_actions

# def fallback_response(question, used_chunks=None):

#     if used_chunks:
#         answer = "I found some related information, but not enough to answer confidently."
#     else:
#         answer = "I couldn’t find exact details, but I can connect you to reception."

#     # 🔥 ADD ACTIONS
#     actions = generate_actions(answer, "fallback", question)

#     return answer, actions

from agent.utils.action_generator import generate_actions

def fallback_response(question, used_chunks=None):

    if used_chunks:
        answer = "I found some related information, but not enough to answer confidently."
    else:
        answer = "I couldn’t find exact details, but I can help you connect with the hotel staff."

    actions = generate_actions(answer, "fallback", question)

    return answer, actions