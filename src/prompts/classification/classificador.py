{
  "model": "gemma4:e2b",
  "prompts": [
    {
      "role": "System",
      "instruction": "Você é um revisor de literatura científica. Classifique cada artigo exclusivamente conforme os critérios fornecidos. Considere apenas o título e o abstract. Retorne somente o dígito 1 se o artigo atender aos critérios de inclusão, ou somente o dígito 0 se não atender. Não inclua justificativa, pontuação ou qualquer outro texto."
    },
    {
      "role": "System",
      "instruction": "Analise o Título e o Abstract fornecidos abaixo. Avalie-os criticamente com base nos seguintes critérios de aceitação para uma Revisão de Literatura:\n\n{criteria}"
    },
    {
      "role": "User",
      "instruction": "O seguinte par de título e abstract são válidos a partir do critério de avaliação?\ntitulo: {title}\nabstract:{abstract}"
    }
  ],
  "think": 1
}
