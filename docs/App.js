import React from "react";
import { FaGithub, FaEnvelope } from "react-icons/fa";

export default function App() {
  return (
    <div className="min-h-screen bg-gray-100 text-gray-900">
      <header className="bg-blue-900 text-white py-6 text-center">
        <h1 className="text-3xl font-bold">Mestrado em Informática - UFAL</h1>
        <p className="text-lg">Fábio Linhares</p>
      </header>
      <main className="max-w-4xl mx-auto p-6">
        <section className="mb-8">
          <h2 className="text-2xl font-semibold text-blue-800">Sobre o Projeto</h2>
          <p className="mt-2 text-gray-700">
            Bem-vindo ao meu repositório de mestrado em Informática na Universidade Federal de Alagoas (UFAL). Aqui compartilho estudos, projetos e materiais desenvolvidos durante o curso.
          </p>
        </section>
        <section className="mb-8">
          <h2 className="text-2xl font-semibold text-blue-800">Conteúdo do Repositório</h2>
          <ul className="mt-2 space-y-2">
            <li><a href="./projetos" className="text-blue-600 hover:underline">Projetos</a></li>
            <li><a href="./artigos" className="text-blue-600 hover:underline">Artigos e Publicações</a></li>
            <li><a href="./apresentacoes" className="text-blue-600 hover:underline">Apresentações</a></li>
            <li><a href="./codigo" className="text-blue-600 hover:underline">Códigos-fonte</a></li>
            <li><a href="./documentos" className="text-blue-600 hover:underline">Documentos de Pesquisa</a></li>
          </ul>
        </section>
        <section className="mb-8">
          <h2 className="text-2xl font-semibold text-blue-800">Áreas de Pesquisa</h2>
          <ul className="mt-2 space-y-2">
            <li className="text-gray-700">Inteligência Artificial</li>
            <li className="text-gray-700">Aprendizado de Máquina</li>
            <li className="text-gray-700">Processamento de Linguagem Natural</li>
          </ul>
        </section>
        <section className="mb-8">
          <h2 className="text-2xl font-semibold text-blue-800">Contato</h2>
          <ul className="mt-2 space-y-2">
            <li className="flex items-center text-gray-700"><FaEnvelope className="mr-2" /> <a href="mailto:fabiolinharez@gmail.com" className="text-blue-600 hover:underline">fabiolinharez@gmail.com</a></li>
            <li className="flex items-center text-gray-700"><FaGithub className="mr-2" /> <a href="https://github.com/fabio-linhares" className="text-blue-600 hover:underline">github.com/fabio-linhares</a></li>
          </ul>
        </section>
      </main>
      <footer className="bg-gray-200 text-center py-4 text-gray-700 text-sm">
        <p>&copy; 2025 Fábio Linhares - Todos os direitos reservados</p>
        <p>
          Este trabalho está licenciado sob a <a href="https://creativecommons.org/licenses/by/4.0/" className="text-blue-600 hover:underline">Licença Creative Commons Attribution 4.0 International</a>.
        </p>
      </footer>
    </div>
  );
}
