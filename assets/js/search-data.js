// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-about",
    title: "about",
    section: "Navigation",
    handler: () => {
      window.location.href = "/portfolio/";
    },
  },{id: "nav-blog",
          title: "blog",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/portfolio/blog/";
          },
        },{id: "nav-projects",
          title: "projects",
          description: "A growing collection of my cool projects.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/portfolio/projects/";
          },
        },{id: "nav-repositories",
          title: "repositories",
          description: "My GitHub repositories",
          section: "Navigation",
          handler: () => {
            window.location.href = "/portfolio/repositories/";
          },
        },{id: "nav-cv",
          title: "cv",
          description: "Below lies my brief CV",
          section: "Navigation",
          handler: () => {
            window.location.href = "/portfolio/cv/";
          },
        },{id: "post-pegasos-kernel-svm-the-hardest-math-so-far",
        
          title: "Pegasos Kernel SVM: The Hardest Math So Far",
        
        description: "Implementing a deterministic Pegasos algorithm for kernel SVMs and diving deep into the mathematics of support vector machines.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/portfolio/blog/2026/pegasos-svm/";
          
        },
      },{id: "post-building-a-primitive-gpt-2-layers-amp-dimensions",
        
          title: "Building a Primitive GPT-2: Layers &amp; Dimensions",
        
        description: "Merging deep learning concepts to build a simplified GPT-2 text generator and understanding the importance of dimensions.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/portfolio/blog/2026/building-gpt2/";
          
        },
      },{id: "post-building-autograd-chain-rule-and-topo-sort",
        
          title: "Building Autograd: Chain Rule and Topo-Sort",
        
        description: "Implementing a basic autograd engine from scratch and discovering the real-world power of topological sorting.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/portfolio/blog/2026/building-autograd/";
          
        },
      },{id: "post-understanding-lstms-a-hands-on-intuition",
        
          title: "Understanding LSTMs: A Hands-on Intuition",
        
        description: "Implementing Long Short-Term Memory (LSTM) from scratch and exploring the intuition behind its gates.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/portfolio/blog/2026/understanding-lstms/";
          
        },
      },{id: "post-earned-deep-ml-badges-attention-is-all-you-need-amp-resnet",
        
          title: "Earned Deep-ML Badges - Attention Is All You Need &amp; ResNet",
        
        description: "Celebrating the completion of Deep-ML problem collections on Transformers and ResNet architectures.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/portfolio/blog/2026/deep-ml-badges/";
          
        },
      },{id: "post-hello-world",
        
          title: "Hello World",
        
        description: "Welcome to my new portfolio website!",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/portfolio/blog/2026/hello-world/";
          
        },
      },{id: "books-the-godfather",
          title: 'The Godfather',
          description: "",
          section: "Books",handler: () => {
              window.location.href = "/portfolio/books/the_godfather/";
            },},{id: "news-a-simple-inline-announcement",
          title: 'A simple inline announcement.',
          description: "",
          section: "News",},{id: "news-a-long-announcement-with-details",
          title: 'A long announcement with details',
          description: "",
          section: "News",handler: () => {
              window.location.href = "/portfolio/news/announcement_2/";
            },},{id: "news-a-simple-inline-announcement-with-markdown-emoji-sparkles-smile",
          title: 'A simple inline announcement with Markdown emoji! :sparkles: :smile:',
          description: "",
          section: "News",},{id: "projects-panic-driven-crowd-dynamics",
          title: 'Panic-Driven Crowd Dynamics',
          description: "Developed strategies to reduce death count in dense crowd panic situations through simulation.",
          section: "Projects",handler: () => {
              window.location.href = "/portfolio/projects/crowd_evacuation/";
            },},{id: "projects-dscoin-cryptocurrency",
          title: 'DSCoin - Cryptocurrency',
          description: "Built cryptocurrency system using SHA256 hashing and implemented mechanism to validate transactions.",
          section: "Projects",handler: () => {
              window.location.href = "/portfolio/projects/dscoin/";
            },},{id: "teachings-data-science-fundamentals",
          title: 'Data Science Fundamentals',
          description: "This course covers the foundational aspects of data science, including data collection, cleaning, analysis, and visualization. Students will learn practical skills for working with real-world datasets.",
          section: "Teachings",handler: () => {
              window.location.href = "/portfolio/teachings/data-science-fundamentals/";
            },},{id: "teachings-introduction-to-machine-learning",
          title: 'Introduction to Machine Learning',
          description: "This course provides an introduction to machine learning concepts, algorithms, and applications. Students will learn about supervised and unsupervised learning, model evaluation, and practical implementations.",
          section: "Teachings",handler: () => {
              window.location.href = "/portfolio/teachings/introduction-to-machine-learning/";
            },},{
        id: 'social-cv',
        title: 'CV',
        section: 'Socials',
        handler: () => {
          window.open("/portfolio/assets/pdf/Yash_CV.pdf", "_blank");
        },
      },{
        id: 'social-email',
        title: 'email',
        section: 'Socials',
        handler: () => {
          window.open("mailto:%79%61%73%68%6D%6A%61%79%73%77%61%6C@%67%6D%61%69%6C.%63%6F%6D", "_blank");
        },
      },{
        id: 'social-github',
        title: 'GitHub',
        section: 'Socials',
        handler: () => {
          window.open("https://github.com/YashJayswal24", "_blank");
        },
      },{
        id: 'social-linkedin',
        title: 'LinkedIn',
        section: 'Socials',
        handler: () => {
          window.open("https://www.linkedin.com/in/yash-jayswal", "_blank");
        },
      },{
        id: 'social-leetcode',
        title: 'LeetCode',
        section: 'Socials',
        handler: () => {
          window.open("https://leetcode.com/u/NemesisX/", "_blank");
        },
      },{
        id: 'social-kaggle',
        title: 'Kaggle',
        section: 'Socials',
        handler: () => {
          window.open("https://www.kaggle.com/reptor420", "_blank");
        },
      },{
        id: 'social-custom_social',
        title: 'Custom_social',
        section: 'Socials',
        handler: () => {
          window.open("https://codeforces.com/profile/nemesis_R", "_blank");
        },
      },{
      id: 'light-theme',
      title: 'Change theme to light',
      description: 'Change the theme of the site to Light',
      section: 'Theme',
      handler: () => {
        setThemeSetting("light");
      },
    },
    {
      id: 'dark-theme',
      title: 'Change theme to dark',
      description: 'Change the theme of the site to Dark',
      section: 'Theme',
      handler: () => {
        setThemeSetting("dark");
      },
    },
    {
      id: 'system-theme',
      title: 'Use system default theme',
      description: 'Change the theme of the site to System Default',
      section: 'Theme',
      handler: () => {
        setThemeSetting("system");
      },
    },];
