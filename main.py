import os
import webbrowser
from dotenv import load_dotenv, find_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Carregar variáveis de ambiente
load_dotenv(find_dotenv())
api_key = os.getenv("GROQ_API_KEY")

# Inicializar agentes ChatGroq
topic_agent = ChatGroq(
    temperature=1,
    api_key=api_key,
    model_name="llama-3.3-70b-versatile"
)

question_agent = ChatGroq(
    temperature=1,
    api_key=api_key,
    model_name="llama-3.3-70b-versatile"
)

text_agent = ChatGroq(
    temperature=1,
    api_key=api_key,
    model_name="llama-3.3-70b-versatile"
)

# Função para gerar tópicos
def generate_topics(excluded_topics=None) -> list:
    excluded_topics = excluded_topics or []
    system_prompt = (
        """Você é um especialista em marketing de conteúdo focado em tecnologia e dados. Sua tarefa é criar uma lista de ideias criativas para posts no LinkedIn sobre os temas de Análise de Dados, Business Intelligence, Data Science e Programação.  

        # Instruções:  
        - Gere 5 ideias de postagens que sejam envolventes, informativas e que gerem engajamento.  
        - Varie os tipos de post: dicas práticas, insights de carreira, tendências da área, desafios comuns, estudos de caso, storytelling pessoal e provocações para discussão.  
        - Considere o público-alvo: analistas, cientistas de dados, desenvolvedores, líderes de tecnologia e entusiastas da área.  
        - Use um tom profissional, mas acessível, com linguagem clara e impactante.  
        - Evite ideias genéricas – traga abordagens diferenciadas e relevantes.  

        # Exemplos de formatos de post:  
        - **Storytelling**: “O que eu gostaria de saber antes de começar na área de {{TEMA}}…”  
        - **Dicas Práticas**: “5 erros comuns que todo iniciante em {{TEMA}} comete (e como evitar)”  
        - **Provocação**: “Será que ainda vale a pena aprender {{LINGUAGEM/TECNOLOGIA}} em {{ANO}}?”  
        - **Tendência**: “O futuro do {{TEMA}}: quais habilidades serão essenciais nos próximos anos?”  
        - **Comparação**: “SQL vs. Python para análise de dados – Qual é melhor para o seu caso?”  

        Gere as ideias de postagens dentro das tags <topicos></topicos>.  

        # Exemplo de saída esperada:  
        <topicos>  
        1. Como começar em Data Science sem experiência (um roadmap prático)  
        2. A verdade sobre certificações em BI: vale a pena investir?  
        3. 3 hacks de Python que todo Analista de Dados deveria conhecer  
        4. Inteligência Artificial vai substituir os Analistas de Dados? Minha visão sobre o futuro da profissão  
        5. De Analista a Gestor de Dados: Como crescer na carreira e liderar times  
        </topicos>  
        """
        f"Não inclua os seguintes tópicos: {', '.join(excluded_topics)}." if excluded_topics else ""
    )
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt)])
    chain = prompt | topic_agent
    response = chain.invoke({"text": ""})
    
    topics = response.content.split("\n")
    return [topic.strip() for topic in topics if topic.strip()]

# Função para gerar perguntas usando o primeiro agente
def generate_prompt_questions(topic: str) -> str:
    system_prompt = """# Instruções

        Você é um especialista em análise técnica e pensamento crítico. Sua tarefa é gerar uma série de perguntas técnicas aprofundadas sobre um determinado tópico, levando em conta diferentes aspectos e implicações. O objetivo é desencadear um pensamento em cadeia, onde cada pergunta aprofunda a anterior ou explora novas direções relacionadas ao tema.

        ## Parâmetros de entrada

        <tópico>  
        {{TÓPICO_FORNECIDO}}  
        </tópico>  

        ## Diretrizes

        1. Comece com uma pergunta técnica fundamental sobre o tópico.  
        2. A partir dessa pergunta, gere perguntas adicionais que aprofundem ou expandam o pensamento.  
        3. Utilize abordagens multidisciplinares quando aplicável.  
        4. Considere impactos práticos, teóricos e futuros sobre o tema.  
        5. As perguntas devem cobrir diferentes níveis de complexidade, indo do básico ao avançado.  

        ## Formato de saída

        As perguntas serão listadas e numeradas em ordem lógica de complexidade e relação com o tópico inicial:

        1. **[Pergunta inicial sobre o tema]**  
        2. **[Pergunta que aprofunda a anterior]**  
        3. **[Pergunta que explora uma perspectiva alternativa]**  
        4. **[Pergunta sobre implicações práticas]**  
        5. **[Pergunta desafiadora que estimula pensamento crítico]**  
        6. **[Pergunta avançada sobre aplicações futuras]**  

        ### Exemplo de saída para o tópico "{{TÓPICO_FORNECIDO}}":  

        1. O que é {{TÓPICO_FORNECIDO}} e como ele funciona?  
        2. Quais são os principais desafios técnicos na implementação de {{TÓPICO_FORNECIDO}}?  
        3. Como a abordagem X difere da abordagem Y na resolução de problemas relacionados a {{TÓPICO_FORNECIDO}}?  
        4. Quais são as aplicações práticas de {{TÓPICO_FORNECIDO}} em diferentes indústrias?  
        5. Quais limitações atuais impedem o avanço de {{TÓPICO_FORNECIDO}} e quais são as possíveis soluções?  
        6. Como {{TÓPICO_FORNECIDO}} pode evoluir nos próximos 10 anos com novas tecnologias emergentes?  
        """
    human_prompt = f"<tópico>  {topic} </tópico> "
    
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", human_prompt)])
    chain = prompt | question_agent
    response = chain.invoke({"text": ""})
    
    return response.content

# Função para gerar texto usando o segundo agente
def generate_text(questions: str, feedback: str = None, texto_anterior: str = None) -> str:
    system_prompt = """# Objetivo  
        Você é um especialista em criação de conteúdo para LinkedIn, ajudando um Analista de BI Pleno a aumentar o engajamento e compartilhar insights valiosos sobre dados, programação e inteligência artificial. Seu tom deve ser jovem, bem-humorado e envolvente, tornando temas técnicos acessíveis e interessantes para suas conexões.  

        ## Informações sobre o Criador de Conteúdo  
        - **Cargo:** Analista de BI Pleno  
        - **Personalidade:** Jovem, descontraído e com um humor sarcastico, apaixonado por tecnologia  
        - **Objetivo:** Aumentar engajamento, atrair conexões de recrutadores e trazer conteúdos relevantes sobre BI, programação e IA  
        - **Tom de voz:** Leve, bem-humorado, envolvente, sem perder a credibilidade  

        ##Lembre-se
        - NÃO USAR MARKDOWNS.
        - Use emoticons sem exagerar.

        ## Estrutura do Post  
        Crie um post seguindo esta estrutura:  

        1. **Abertura Impactante** – Comece com uma frase chamativa, uma pergunta instigante ou uma piada relacionada ao mundo dos dados e tecnologia. Algo que faça as pessoas pararem para ler.  
        2. **Contexto e Reflexão** – Introduza o tema do post com uma situação do dia a dia, uma tendência de mercado ou um aprendizado pessoal, conectando-se com a realidade da audiência.  
        3. **Conteúdo Principal** – Explique um conceito de BI, programação ou IA de forma simples e envolvente, usando analogias, memes ou exemplos práticos.  
        4. **Chamado à Interação** – Termine o post incentivando comentários e discussões, seja pedindo opinião, sugerindo um desafio ou criando um debate amigável.  

        ## Exemplos de Tópicos  
        Aqui estão algumas ideias de posts que podem ser criados seguindo essa estrutura:  

        - "O dia em que meu SQL rodou mais devagar que meu Wi-Fi de madrugada… e o que aprendi com isso!"  
        - "Se o Excel fosse um super-herói, qual seria seu superpoder?"  
        - "Afinal, será que a IA vai roubar o nosso trampo ou só transformar a gente em super-hackers dos dados?"  
        - "5 atalhos no Python que vão te fazer parecer um mago da programação 🧙‍♂️"  
        - "O que o ChatGPT pode (e não pode) fazer por um Analista de BI?"  

        ## Formato da Resposta  
        O post final deve ser entregue dentro das tags `<post> </post>` e seguir a estrutura mencionada.  

        ### Exemplo de saída esperada:  

        ```xml
        <post>
        🚀 O SQL rodou… mas levou tanto tempo que pensei em trocar de carreira  

        Quem nunca escreveu uma query e foi tomar um café esperando o resultado? 🫠  
        Hoje vou compartilhar 3 otimizações simples que podem salvar seu tempo (e seu café):  

        1️⃣ Use índices para acelerar buscas  
        2️⃣ Evite SELECT * – só pegue o que precisa  
        3️⃣ Subqueries? Prefira CTEs!  

        Já testou essas dicas? Conta aí nos comentários se seu SQL já te fez sofrer! 😂  
        </post>

        ##Lembre-se
        - NÃO USAR MARKDOWNS.
        - Use emoticons sem exagerar.
        """
    if feedback:
        response_user = (
            f"#Aplique o seguinte feedback:\n {feedback}" 
            f"#Aplique no texto a seguir:\n {texto_anterior}"
        )
        system_prompt = response_user
    
    human_prompt = f"Sinta se a vontade para responder e pensar sobre as seguintes questões antes de responder: {questions}"
    
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", human_prompt)])
    chain = prompt | text_agent
    response = chain.invoke({"text": ""})
    
    return response.content

# Função para salvar texto em arquivo e abrir navegador
def save_and_open_text_file(content: str, filename: str = "linkedin_post.txt"):
    with open(filename, "w", encoding="utf-8") as file:
        file.write(content)
    
    os.system(f"start {filename}" if os.name == "nt" else f"open {filename}")
    webbrowser.open("https://www.linkedin.com")

# Fluxo principal
def main():
    try:
        excluded_topics = []
        while True:
            print("Escolha uma opção:")
            print("1. Gerar tópicos automaticamente")
            print("2. Inserir um tópico manualmente")
            option = input("Opção escolhida (1/2): ").strip()
            
            if option == '1':
                print("Gerando tópicos...")
                topics = generate_topics(excluded_topics)
                
                print("\nTópicos sugeridos:")
                for i, topic in enumerate(topics, start=1):
                    print(f"{i}. {topic}")
                
                choice = input("\nEscolha o número do tópico desejado ou 'R' para gerar novos tópicos: ").strip().upper()
                if choice == 'R':
                    excluded_topics.extend(topics)
                    print("\nGerando novos tópicos excluindo os anteriores...\n")
                    continue
                
                if choice.isdigit() and 1 <= int(choice) <= len(topics):
                    selected_topic = topics[int(choice) - 1]
                else:
                    print("Opção inválida. Tente novamente.")
                    continue
            elif option == '2':
                selected_topic = input("Insira o tópico desejado: ").strip()
            else:
                print("Opção inválida. Tente novamente.")
                continue
            
            print(f"\nTópico escolhido: {selected_topic}")
            
            # Gerar perguntas
            print("\nGerando perguntas...")
            questions = generate_prompt_questions(selected_topic)
            print(f"Perguntas geradas:\n{questions}\n")
            
            # Gerar texto final
            while True:
                print("Gerando texto final...")
                linkedin_text = generate_text(questions)
                print(f"Texto gerado:\n{linkedin_text}\n")
                
                feedback = input("Gostou do texto? Se não, insira comentários para melhorias ou pressione Enter para aceitar: ").strip()
                if not feedback:
                    break
                print("Reaplicando melhorias com base no feedback...\n")
                linkedin_text = generate_text(questions, feedback, linkedin_text)
            
            # Salvar e abrir o texto
            save_and_open_text_file(linkedin_text)
            print("Texto salvo e pronto para revisão.")
            break
    except Exception as e:
        print(f"Erro durante a execução: {str(e)}")

if __name__ == "__main__":
    main()
