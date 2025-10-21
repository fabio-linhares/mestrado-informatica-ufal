const Header = () => (
  <header className="bg-gradient-to-r from-blue-800 to-blue-600 text-white py-4 shadow-lg">
    <div className="container mx-auto px-4">
      <h1 className="text-3xl font-bold">Mestrado em Informática - UFAL</h1>
      <p className="text-lg">Fábio Linhares</p>
    </div>
  </header>
);

const Navigation = () => (
  <nav className="bg-gray-100 shadow-md">
    <div className="container mx-auto px-4">
      <ul className="flex space-x-4 py-3">
        <li><a href="#sobre" className="text-blue-600 hover:text-blue-800 transition duration-300">Sobre</a></li>
        <li><a href="#conteudo" className="text-blue-600 hover:text-blue-800 transition duration-300">Conteúdo</a></li>
        <li><a href="#pesquisa" className="text-blue-600 hover:text-blue-800 transition duration-300">Pesquisa</a></li>
        <li><a href="#contato" className="text-blue-600 hover:text-blue-800 transition duration-300">Contato</a></li>
      </ul>
    </div>
  </nav>
);

const Section = ({ id, title, children }) => (
  <section id={id} className="mb-12">
    <h2 className="text-2xl font-semibold text-blue-800 mb-4 border-b-2 border-blue-200 pb-2">{title}</h2>
    {children}
  </section>
);

const FaGithub = () => (
  <svg viewBox="0 0 496 512" fill="currentColor" className="w-5 h-5 inline-block mr-2">
    <path d="M165.9 397.4c0 2-2.3 3.6-5.2 3.6-3.3.3-5.6-1.3-5.6-3.6 0-2 2.3-3.6 5.2-3.6 3-.3 5.6 1.3 5.6 3.6zm-31.1-4.5c-.7 2 1.3 4.3 4.3 4.9 2.6 1 5.6 0 6.2-2s-1.3-4.3-4.3-5.2c-2.6-.7-5.5.3-6.2 2.3zm44.2-1.7c-2.9.7-4.9 2.6-4.6 4.9.3 2 2.9 3.3 5.9 2.6 2.9-.7 4.9-2.6 4.6-4.6-.3-1.9-3-3.2-5.9-2.9zM244.8 8C106.1 8 0 113.3 0 252c0 110.9 69.8 205.8 169.5 239.2 12.8 2.3 17.3-5.6 17.3-12.1 0-6.2-.3-40.4-.3-61.4 0 0-70 15-84.7-29.8 0 0-11.4-29.1-27.8-36.6 0 0-22.9-15.7 1.6-15.4 0 0 24.9 2 38.6 25.8 21.9 38.6 58.6 27.5 72.9 20.9 2.3-16 8.8-27.1 16-33.7-55.9-6.2-112.3-14.3-112.3-110.5 0-27.5 7.6-41.3 23.6-58.9-2.6-6.5-11.1-33.3 2.6-67.9 20.9-6.5 69 27 69 27 20-5.6 41.5-8.5 62.8-8.5s42.8 2.9 62.8 8.5c0 0 48.1-33.6 69-27 13.7 34.7 5.2 61.4 2.6 67.9 16 17.7 25.8 31.5 25.8 58.9 0 96.5-58.9 104.2-114.8 110.5 9.2 7.9 17 22.9 17 46.4 0 33.7-.3 75.4-.3 83.6 0 6.5 4.6 14.4 17.3 12.1C428.2 457.8 496 362.9 496 252 496 113.3 383.5 8 244.8 8zM97.2 352.9c-1.3 1-1 3.3.7 5.2 1.6 1.6 3.9 2.3 5.2 1 1.3-1 1-3.3-.7-5.2-1.6-1.6-3.9-2.3-5.2-1zm-10.8-8.1c-.7 1.3.3 2.9 2.3 3.9 1.6 1 3.6.7 4.3-.7.7-1.3-.3-2.9-2.3-3.9-2-.6-3.6-.3-4.3.7zm32.4 35.6c-1.6 1.3-1 4.3 1.3 6.2 2.3 2.3 5.2 2.6 6.5 1 1.3-1.3.7-4.3-1.3-6.2-2.2-2.3-5.2-2.6-6.5-1zm-11.4-14.7c-1.6 1-1.6 3.6 0 5.9 1.6 2.3 4.3 3.3 5.6 2.3 1.6-1.3 1.6-3.9 0-6.2-1.4-2.3-4-3.3-5.6-2z" />
  </svg>
);

const FaEnvelope = () => (
  <svg viewBox="0 0 512 512" fill="currentColor" className="w-5 h-5 inline-block mr-2">
    <path d="M48 64C21.5 64 0 85.5 0 112c0 15.1 7.1 29.3 19.2 38.4L236.8 313.6c11.4 8.5 27 8.5 38.4 0L492.8 150.4c12.1-9.1 19.2-23.3 19.2-38.4c0-26.5-21.5-48-48-48H48zM0 176V384c0 35.3 28.7 64 64 64H448c35.3 0 64-28.7 64-64V176L294.4 339.2c-22.8 17.1-54 17.1-76.8 0L0 176z" />
  </svg>
);

function App() {
  return (
    <div className="min-h-screen bg-gray-50 text-gray-900 flex flex-col">
      <Header />
      <Navigation />
      <main className="flex-grow container mx-auto px-4 py-8">
        <Section id="sobre" title="Sobre o Projeto">
          <p className="text-gray-700 leading-relaxed">
            Bem-vindo ao meu repositório de mestrado em Informática na Universidade Federal de Alagoas (UFAL). 
            Aqui compartilho estudos, projetos e materiais desenvolvidos durante o curso, focando em áreas 
            como Inteligência Artificial, Aprendizado de Máquina e Processamento de Linguagem Natural.
          </p>
        </Section>

        <Section id="conteudo" title="Conteúdo do Repositório">
          <ul className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {[
              { name: "Projetos", newTab: false, href: "#" },
              { name: "Artigos e Publicações", newTab: false, href: "#" },
              { name: "Apresentações", newTab: true, href: "./apresentacoes.html" },
              { name: "Códigos-fonte", newTab: false, href: "#" },
              { name: "Documentos de Pesquisa", newTab: false, href: "#" },  
              { name: "Disciplinas", newTab: false, href: "./disciplinas.html" }
            ].map((item, index) => (
              <li key={index} className="bg-white p-4 rounded-lg shadow-md hover:shadow-lg transition duration-300">
                <a 
                  href={item.href} 
                  className="text-blue-600 hover:text-blue-800 transition duration-300"
                  {...(item.newTab ? { target: "_blank", rel: "noopener noreferrer" } : {})}
                >
                  {item.name}
                </a>
              </li>
            ))}
          </ul>
        </Section>

        <Section id="pesquisa" title="Áreas de Pesquisa">
          <ul className="list-disc list-inside space-y-2 text-gray-700">
            <li>Inteligência Artificial</li>
            <li>Aprendizado de Máquina</li>
            <li>Processamento de Linguagem Natural</li>
            <li>Visão Computacional</li>
          </ul>
        </Section>

        <Section id="contato" title="Contato">
          <ul className="space-y-2">
            <li className="flex items-center text-gray-700">
              <FaEnvelope />
              <a href="mailto:fabiolinharez@gmail.com" className="text-blue-600 hover:text-blue-800 transition duration-300">
                fabiolinharez@gmail.com
              </a>
            </li>
            <li className="flex items-center text-gray-700">
              <FaGithub />
              <a href="https://github.com/fabio-linhares" className="text-blue-600 hover:text-blue-800 transition duration-300">
                github.com/fabio-linhares
              </a>
            </li>
          </ul>
        </Section>
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