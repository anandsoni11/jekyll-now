---
layout: default
title: Bits
permalink: /
feature-img: "img/sample_feature_img_2.png"
---

<div class="home">
  {% if site.theme.header_text %}
  <div class="call-out" 
  style="background-image: url('{{ site.baseurl }}/{{ site.theme.header_text_feature_image }}')">
    {{ site.theme.header_text }}
  </div>
  {% endif %}

  <div class="posts">
    {% for post in site.categories.binary %}
    <div class="post-teaser">
      <header>
        <h1>
          <a class="post-link" href="{{ post.url | prepend: site.baseurl }}">
            {{ post.title }}
          </a>
        </h1>
        <p class="meta">
          {{ post.date | date: "%B %-d, %Y" }}
        </p>
      </header>
      <div class="excerpt">
        {{ post.excerpt }}
        <a class="button" href="{{ post.url | prepend: site.baseurl }}">
          {{ site.theme.str_continue_reading }}
        </a>
      </div>
    </div>
    {% endfor %}
  </div>

  {% if paginator.total_pages > 1 %}
  <div class="pagination">
    {% if paginator.previous_page %}
    <a href="{{ paginator.previous_page_path | prepend: site.baseurl | replace: '//', '/' }}" class="button" >
      <i class="fa fa-chevron-left"></i>
      {{ site.theme.str_prev }}
    </a>
    {% endif %}
    {% if paginator.next_page %}
    <a href="{{ paginator.next_page_path | prepend: site.baseurl | replace: '//', '/' }}" class="button" >
      {{ site.theme.str_next }}
      <i class="fa fa-chevron-right"></i>
    </a>
    {% endif %}
  </div>
  {% endif %}
</div>