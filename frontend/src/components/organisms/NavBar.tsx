import { SvgIcon } from "../atoms/SvgIcon";
import { useNavigate } from "react-router-dom";

export default function NavBar() {
  const navigate = useNavigate();

  return (
    <nav
      className="sticky flex flex-col items-center justify-between border-r border-white border-opacity-50 px-2 py-3 
        max-md:flex-row max-md:border-b max-md:px-3 max-md:py-2 md:min-h-screen"
    >
      <SvgIcon
        type="logo"
        className="h-6 w-6 cursor-pointer"
        onClick={() => navigate("/app")}
      />
      <div className="flex flex-col items-center max-md:flex-row max-md:space-x-3 md:space-y-3">
        <a href={"/app"} target="_blank" rel="noopener noreferrer">
          <SvgIcon type="donation" className="h-5 w-5" />
        </a>
        <a
          href={"https://hub.docker.com/u/ilovedatajjia"}
          target="_blank"
          rel="noopener noreferrer"
        >
          <SvgIcon type="docker" className="h-5 w-5" />
        </a>
        <a
          href={"https://github.com/iLoveDataJjia"}
          target="_blank"
          rel="noopener noreferrer"
        >
          <SvgIcon type="github" className="h-5 w-5" />
        </a>
      </div>
    </nav>
  );
}
